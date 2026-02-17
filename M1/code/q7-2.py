# We need a table to hold the value for every unique state
# 8 Rooms * 25 Tiles = 200 Values
total_values = np.zeros((8, 25))

def update_values():
    # Loop forever until values stop changing
    while True:
        delta = 0
        # Loop through every room (Macro)
        for room_id in range(8):
            # Loop through every tile (Micro)
            for tile_id in range(25):

                # 1. Ask the generic "Lego" logic where we go
                next_room, next_tile = get_next_state(room_id, tile_id)

                # 2. Look up the value of that destination
                # (Notice we are reading from the specific 'total_values' table)
                dest_value = total_values[next_room, next_tile]

                # 3. Update current state
                new_val = Reward + gamma * dest_value
                total_values[room_id, tile_id] = new_val