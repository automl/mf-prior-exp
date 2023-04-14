import os

# Specify the base directory path
base_dir = "/path/to/base/folder"

# Specify the string to be replaced and the new string
old_string = "pb_mutation_decay_linear-default-at-target"  # Replace with the actual string to be replaced
new_string = "pb_mutation_dynamic_linear-default-at-target"  # Replace with the actual new string

# Get all the folders at the base level
base_level_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

# Loop through each base level folder
for folder in base_level_folders:
    if "cluster_oe" in folder:
        continue
    folder_path = os.path.join(base_dir, folder)
    # Check if the folder contains sub-folders
    if os.path.isdir(folder_path):
        # Get all the sub-folders in the folder
        sub_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        # Loop through each sub-folder and rename it if it matches the old string
        for sub_folder in sub_folders:
            if old_string in sub_folder:
                print(sub_folder)
                old_sub_folder_path = os.path.join(folder_path, sub_folder)
                new_sub_folder_name = sub_folder.replace(old_string, new_string)
                new_sub_folder_path = os.path.join(folder_path, new_sub_folder_name)
                os.rename(old_sub_folder_path, new_sub_folder_path)
