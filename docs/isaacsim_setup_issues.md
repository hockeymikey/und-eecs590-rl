# Isaac Lab — Container Setup Notes

These are the notes and lessons from my journey getting Isaac Lab v2.3.2 running or should I say *rerunning* on Arch linux.

Arch linux is a special little beast when it comes to the Isaac set of libraries/applications. They do not have support thus you're
really limited to running it in docker. You could try to build it, but as someone who tried that, I wouldn't recommend it.
I had both containers of Isaac Sim and Isaac Lab running on this machine before and used prior, but it seems recent Arch updates 
broke them. This document notes and comments on what worked for me, and can help inspire direct others in similar situation.s

Isaac Lab tag: `v2.3.2`. Isaac Sim base image: `5.1.0`.
Host: Arch Linux, NVIDIA driver 595.71.05 (CUDA 13.2), Docker w/ NVIDIA
Container Toolkit. CUDA 13.2 on host is backward-compatible with the
CUDA 12.8 stack inside Isaac Sim 5.1.0; no driver issues observed.

## TL;DR — the install sequence that worked

After `python docker/container.py start base --files docker-compose.local.yaml`
and entering the container, inside `/workspace/isaaclab`:

```bash
# Upstream pinning workaround for the flatdict build (see "Flatdict failure" below)
isaaclab -p -m pip install pip==23
isaaclab -p -m pip install setuptools==65
isaaclab -p -m pip install flatdict

# Now the full isaaclab install completes
./isaaclab.sh --install
```

Then install this repo inside the container:

```bash
cd /workspace/Minis
isaaclab -p -m pip install -e .[sb3]
```

Verification — bare `import isaaclab` works:
```bash
isaaclab -p -c "import isaaclab; print(isaaclab.__version__)"
# 0.54.2
```

Verification — full stack works (requires Kit boot for pxr; see below):
```bash
isaaclab -p -c "
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import isaaclab.envs
from pxr import Usd
import rl590.envs.zamb_isaac
print('full stack ok')
app.close()
"
```

## Issues encountered, in the order they bit

### 1. `docker-compose.local.yaml` not auto-loaded

The file is a template. `container.py` only loads `docker-compose.yaml` by
default. To merge the local file:

```
python docker/container.py start base --files docker-compose.local.yaml
```

Same `--files` flag needed on every container.py subcommand (`stop`,
`enter`, `config`).

### 2. Service name mismatch in `docker-compose.local.yaml`

The template comments say "The 'base' service is what is launched" — that's
misleading. `base` is the **profile**; the actual service in
`docker-compose.yaml` is named `isaac-lab-base`. Overrides must use the
service name, not the profile name:

```yaml
services:
  isaac-lab-base:    # NOT "base"
    volumes:
      - ~/local-project-folder:/workspace/local-project-folder:rw
```

### 3. Python-version pin mismatch

Isaac Sim 5.1.0's bundled Python is 3.11. Any package installed into the
Isaac container must allow Python 3.11 even if the normal gym-side workflow
uses Python 3.12. None of the Isaac-side code uses 3.12-only syntax.

### 4. numpy 2.x vs Isaac Lab's `numpy<2` pin

`isaaclab_tasks/setup.py` hard-pins `numpy<2`. If a local editable package
requires `numpy>=2.0`, pip can resolve the inequality by installing numpy 2.x
and breaking Isaac Lab silently.

**Fix**: keep Isaac-container installs compatible with `numpy>=1.26,<2.0`;
revisit once Isaac Lab's 3.x line drops the `<2` pin.
None of our code uses numpy 2-only APIs.

This is forward-looking pessimism: Isaac Lab 3.0+ moves to numpy 2, but
isn't usable yet (3.0.0 references `nvcr.io/nvidia/isaac-sim:6.0.0` which
isn't published on NGC).

### 5. The flatdict build failure — the real wall

The `isaaclab` main package depends on `flatdict==4.0.1`, an ancient PyPI
package (last release 2019). When pip builds flatdict in its isolated build
environment, the build fails with:

```
ModuleNotFoundError: No module named 'pkg_resources'
```

Cause: modern pip + modern setuptools removed `pkg_resources` from the
isolated build env's defaults. flatdict's `setup.py` predates this and
still relies on it.

The failure aborts the `isaaclab` package install but **does not abort the
overall `./isaaclab.sh --install` script** — subpackages (`isaaclab_rl`,
`isaaclab_tasks`, `isaaclab_assets`, `isaaclab_mimic`) continue installing
successfully. End result: `import isaaclab` fails while subpackages exist.
Surprisingly hard to debug because the script "succeeds."

**Fix (applied)**: downgrade pip + setuptools, pre-install flatdict before
running `--install`. The downgraded pip uses an older isolated build env
where `pkg_resources` is still available.

```bash
isaaclab -p -m pip install pip==23
isaaclab -p -m pip install setuptools==65
isaaclab -p -m pip install flatdict
```

**Upstream tracking**:
- https://github.com/isaac-sim/IsaacLab/issues/4577 — flatdict build failure
- https://github.com/isaac-sim/IsaacLab/issues/4576 — proposal to remove
  flatdict dependency entirely

Once the flatdict PR lands in a future Isaac Lab release, this workaround
goes away.

### 6. pxr lives in Kit extensions, not site-packages (Isaac Sim 5.1.0)

Isaac Sim 4.5 shipped `pxr` in `_isaac_sim/kit/python/lib/python3.11/site-packages/pxr/`.
Isaac Sim 5.1.0 moved it to `/isaac-sim/extscache/omni.usd.libs-1.0.1+.../pxr/`.

This means `from pxr import Usd` does **not** work in a bare
`python.sh -c` invocation — pxr is only on sys.path after Kit boots via
`AppLauncher`. The top-of-module `from pxr import Usd, UsdGeom` in
`isaaclab/utils/mesh.py` enforces this: `import isaaclab.envs` will
always fail at module load time, **even with a working install**, unless
Kit is booted first.

This isn't a bug we can fix from the environment side. The convention is:
**every Isaac Lab entry-point script bootstraps Kit before importing
isaaclab.envs**. The smoke and training scripts in this repo already do this.
A bare `python -c "import isaaclab.envs"` is the wrong test for this Isaac Sim
version.

The right diagnostic test:

```bash
isaaclab -p -c "
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import isaaclab.envs
from pxr import Usd
print('ok')
app.close()
"
```

If that succeeds, the stack is wired up correctly even if bare
`import isaaclab.envs` errors.

### 7. `rl-games` / `psutil` conflict — cosmetic, ignored

`isaaclab_mimic` pulls `robomimic` which pulls `psutil>=any` and ends up
resolving to psutil 7.2.2. `rl-games` (also in the isaaclab deps) pins
`psutil<6.0.0`. Pip emits:

```
rl-games 1.6.1 requires psutil<6.0.0,>=5.9.0, but you have psutil 7.2.2
which is incompatible.
```

The submitted training paths use SB3 or `rl590.deep.ppo`, so this conflict
never gets exercised at runtime.

### 8. Bind-mount + editable-install footgun

`docker-compose.yaml` bind-mounts `source/`, `scripts/`, `docs/`, `tools/`
from the host into the container. The Dockerfile runs
`./isaaclab.sh --install` at build time, which creates `.egg-info`
directories *inside* `source/` (because that's where editable installs put
metadata). At runtime, the bind mount can shadow build-time `.egg-info`s
if they don't match the host's `source/` state.

If you switch git branches on the host while a container is built against
a different branch, the install metadata in site-packages can point at
paths that no longer match the bind-mounted source tree. The fix is to
re-run the install at runtime inside the container so metadata lands on
the host `source/` and persists correctly:

```bash
cd /workspace/isaaclab
./isaaclab.sh --install
```

**Lesson**: pick your Isaac Lab tag *before* starting the container, and
don't switch tags while the container is alive.

### 9. NVIDIA Vulkan ICD not auto-mounted by Container Toolkit 1.19

Symptom inside the container:

```
[carb.graphics-vulkan.plugin] vkCreateInstance failed.
    Vulkan 1.1 is not supported, or your driver requires an update.
```

Plus:
- `/usr/share/vulkan/icd.d/` does not exist
- `libGLX_nvidia*` not present in `/usr/lib/x86_64-linux-gnu/`
- `nvidia-smi` works (CUDA compute is fine)
- `NVIDIA_DRIVER_CAPABILITIES=all` is set in the container env

NVIDIA Container Toolkit 1.19.x on Arch (driver 595.71.05) **mounts the
NVIDIA compute libs but skips the Vulkan ICD JSON and graphics libs** even
when `NVIDIA_DRIVER_CAPABILITIES=all` is requested. The toolkit's auto-mount
logic has a path-discovery mismatch with how Arch's `nvidia-utils` package
lays out files.

**Fix (applied)**: explicit bind-mount in
`docker-compose.local.yaml`:

```yaml
services:
  isaac-lab-base:
    volumes:
      - ~/path/to/Minis:/workspace/Minis:rw
      - /usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json:ro
      - /usr/share/glvnd/egl_vendor.d/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro
```

After this, `vkCreateInstance` succeeds and Isaac Sim's full extension load
completes. Verify inside container:

```bash
ls /usr/share/vulkan/icd.d/nvidia_icd.json   # must exist
ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia*  # toolkit usually mounts these
```

The `.so` files are usually mounted correctly by the toolkit; it's the
ICD JSON that gets dropped. Once the JSON is in place, the loader walks
the lib paths and finds the driver.

If you ever see `vkCreateInstance failed` errors again, this bind-mount
is the first thing to re-verify. I suspect this error was due to the newer drivers and not using 580.X ones.

### 10. `librtx.scenedb.plugin.so` segfault on RTX 3060 Laptop + Isaac Sim 5.1.0-rc.19

Symptom: Isaac Sim GUI loads all extensions, reaches
`Isaac Sim Full Version: 5.1.0-rc.19`, then crashes during renderer init
right after `SetLightingMenuModeCommand(lighting_mode=stage)`. Stack
trace:

```
librtx.scenedb.plugin.so!carbOnPluginStartup+0x3b4de
  std::vector<std::tuple<char const*, float, float, unsigned int, unsigned int, unsigned int>>::_M_realloc_insert
libcarb.scenerenderer-rtx.plugin.so!carbOnPluginShutdown
libomni.hydra.rtx.plugin.so!+0x4459
```

This is a real upstream bug in the RTX scene-database plugin shipped with
Isaac Sim 5.1.0-rc.19 (note: NGC labels its release-candidate build as
"5.1.0", which is confusing — confirmed via
`lib_isaacSim_buildVersion = '5.1.0-rc.19'` in the crash dump).

The plugin appears to enumerate something (GPU caps? ray-tracing features?)
into a `std::vector` and hits invalid memory. Specific to:
- RTX 3060 Laptop GPU (Ampere, sm_86, 6 GB VRAM)
- NVIDIA driver 595.71.05 (newer than 565.x that 5.1 RCs were tested against)
- 6 GB VRAM is below Isaac Sim's "recommended" 8 GB (note: 2060m with 6 GB
  worked fine on earlier Isaac Sim versions — this is not a hard limit)
- PCIe link reports Gen 1 x8 (max Gen 3 x16) — probably normal idle
  power-management; GPU should boost on engagement.

**GUI workaround attempts that didn't help**:
- Clearing the Omniverse shader cache
- `--/rtx/rendermode=RaytracedLighting`
- `--/app/renderer/active=Storm --/rtx.enabled=false`
- `/isaac-sim/isaac-sim.streaming.sh`
- Vulkan device-selection env vars
- Single-threaded scheduler flags

### RESOLVED 2026-05-12: NVIDIA driver 595.x is unvalidated for Isaac Sim 5.1

After exhausting workarounds (renderer flags, LTS kernel boot, container
rebuilds), the root cause was confirmed via a github issue thread on the
Isaac Sim repo: **NVIDIA driver 595.x is not validated for Isaac Sim 5.1.**
The bug reproduces across many GPU models on both Windows and Linux (RTX
3060 Laptop, 4070, 5070 Ti, 5080, 5090 reported) — it is *not* hardware specific to this 3060 Laptop, despite being below the recommended requirements, 2060m worked fine in the prior build before the Omen 15 2022 switch over.

NVIDIA's recommended fixes (from the issue thread):
- **Linux**: downgrade to driver `580.65.06`.
- **Windows**: downgrade to driver `580.88`.
- **Alternative** (workstation users who can't reboot): disable Vulkan in
  `isaacsim.exp.full.kit` and `isaacsim.exp.base.kit`. On Windows this
  falls back to D3D12. We don't have DirectX on Linux so obviously this isn't a solution.

**Fix applied here**: driver downgrade to `580.159.03` via AUR:
`yay -S nvidia-580xx-dkms`

We also downgraded to the LTS kernel (6.18.29) vs stable's 7.0.5. It's unclear if this is needed, further testing should be done to investigate, but works as is.

### Post-downgrade gotcha: NVIDIA Container Toolkit CDI cache

After the driver downgrade, the first `python docker/container.py start
base --files docker-compose.local.yaml` failed with:

```
OCI runtime create failed ... open /usr/lib/libEGL_nvidia.so.595.71.05:
no such file or directory
```

NVIDIA Container Toolkit (1.19.x) had cached the previous driver's library
list in its CDI (Container Device Interface) spec and was still trying to
bind-mount the `.595.71.05` filenames into the container. The driver
files had moved to `.580.159.03` versions on disk.

**Fix**: regenerate the CDI spec and restart the docker daemon:

```fish
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo systemctl restart docker
```

If `/etc/cdi/nvidia.yaml` doesn't exist after the generate (depends on the
toolkit's discovery mode), force legacy mode which discovers libs at
container-start time:

```fish
sudo nvidia-ctk config --set nvidia-container-runtime.mode=legacy --in-place
sudo systemctl restart docker
```

After either path, the container starts cleanly and `nvidia-smi` inside
it reports the same 580.159.03 as the host.

### Lessons learned from this saga

1. **Driver validation matters more than expected.** Isaac Sim 5.1
   genuinely doesn't work on driver 595. There's no graceful degradation
   or error message pointing at the driver — you get a null-pointer
   segfault in a closed-source plugin. Without finding the upstream issue
   thread, this would have taken many more hours.
2. **Pin your NVIDIA driver on Arch.** Rolling distros + bleeding-edge
   driver updates + closed-source binary plugins compiled against
   specific driver versions = recipe for sudden breakage. Pinning is the
   only reliable workflow.
3. **The Vulkan ICD bind-mount workaround is independent of the driver
   issue.** It addresses NVIDIA Container Toolkit 1.19's failure to
   auto-mount `nvidia_icd.json` on Arch. Keep it in
   `docker-compose.local.yaml` regardless of driver version.
4. **CDI cache invalidation is a thing.** Any host-side driver change
   requires regenerating the CDI spec or switching to legacy discovery
   mode. Otherwise the toolkit tries to mount paths that no longer exist.
5. **The 3060 Laptop / mobile-Ampere / 6 GB VRAM / Arch / new-kernel
   hypotheses were all red herrings.** Reportedly the same crash hits
   workstation 5090s. When chasing a bug on niche hardware, search for
   reports on non-niche hardware too — the bug may not be hardware-axis
   at all.
6. **The headless workaround was real, even if architecturally
   confused.** Bare `AppLauncher(headless=True) + import isaaclab` worked
   throughout.

### What to track for in the future

- NVIDIA publishes a 595.x or 600.x driver *validated for Sim 5.1*. The
  Isaac Sim system requirements page would list it. At that point,
  unpin and roll forward.
- NVIDIA releases Isaac Sim 5.2 / 6.0 that rebuilds the scenedb plugin
  against newer driver headers — could unbreak 595.x.
- Isaac Lab v3.0.0 (out of beta) lands with proper Sim 6.0 support.

### Upstream references

- Closing comment on the github issue describing the 595.x driver
  incompatibility — confirmed root cause for this 6ish hour debugging
  session.
- Isaac Sim 5.1 system requirements page (lists validated drivers).

## The Vulkan ICD bind-mount in summary

The most useful change for *anyone* trying Isaac Sim on Arch with recent
NVIDIA Container Toolkit:

```yaml
# docker-compose.local.yaml
- /usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json:ro
- /usr/share/glvnd/egl_vendor.d/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro
```

This is the single fix that's most likely to be load-bearing for future
container debugging — without it, no graphics application running in
the container can initialize Vulkan, and the error message
("`Vulkan 1.1 is not supported`") will mislead you toward driver issues
that aren't really there.

## Container lifecycle

Inside the container, the alias `isaaclab` (no `.sh`) resolves to
`/workspace/isaaclab/isaaclab.sh`. The bashrc sets it. If `isaaclab`
isn't found, source the bashrc or use the full path.

To rebuild from scratch (only do this if the image is in a known-broken
state, since it costs ~30 min):

```fish
cd ~/Dev/docker/IsaacLab
python docker/container.py stop base --files docker-compose.local.yaml
docker image rm isaac-lab-base 2>/dev/null
docker volume ls --format '{{.Name}}' | grep -iE 'isaac' | xargs -r docker volume rm
docker builder prune -af
rm -rf .                  # only if also re-cloning IsaacLab
git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git .
# Re-create docker-compose.local.yaml
python docker/container.py start base --files docker-compose.local.yaml
```

The `docker_isaac-cache-pip` volume is the most valuable cache — pip
downloads can take 20+ minutes on a cold rebuild without it. Don't remove
it unless you suspect pip-cache poisoning.

## References

Upstream Isaac Lab issues encountered:
- https://github.com/isaac-sim/IsaacLab/issues/4576 — flatdict removal
- https://github.com/isaac-sim/IsaacLab/issues/4577 — flatdict build failure

Isaac Sim 5.1.0 release notes (Kit extension reorg) — search NVIDIA docs
for "Isaac Sim 5.0 migration" / "pxr extension".
