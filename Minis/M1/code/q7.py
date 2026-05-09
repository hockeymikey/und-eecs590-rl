class TileTemplate:
    """Represents a single 5x5 room logic"""
    def __init__(self, size=5):
        self.size = size
        # We only store the logic for ONE room
        # P_local is small (25x25)
        self.P_local = self._build_local_transitions(size)

    def get_next_state(self, current_xy, action):
        """Calculates next local position. Returns 'EXIT' if walking out a door."""
        # ... logic to check walls and move ...
        pass

class TreeWorld:
    """Connects tiles via a Binary Tree"""
    def __init__(self, height=3):
        # Create the tree structure (e.g., Node 1 -> [Node 2, Node 3])
        self.num_nodes = 2**height - 1
        self.tree_structure = self._build_binary_tree(height)

        # The 'Lego Brick' - we reuse this object for ALL rooms!
        # This saves massive memory.
        self.shared_tile = TileTemplate(size=5)

    def transition(self, global_state, action):
        """
        Global State = (Room_ID, Local_X, Local_Y)
        """
        room_id, x, y = global_state

        # 1. Ask the Tile where we move locally
        next_local = self.shared_tile.get_next_state((x,y), action)

        # 2. Handle Room Transitions (The Tree Logic)
        if next_local == "EXIT_LEFT":
            new_room = self.tree_structure[room_id]['left_child']
            return (new_room, 4, y) # Enter at right side of new room
        elif next_local == "EXIT_RIGHT":
            new_room = self.tree_structure[room_id]['right_child']
            return (new_room, 0, y) # Enter at left side of new room

        # 3. Standard move
        return (room_id, next_local[0], next_local[1])