import os

# Define the parent directory. '.' refers to the current directory
# where this script is running.
parent_dir = '.'

# A list of the directory names you want to check.
dir_names = [str(i) for i in range(1, 10)] # This creates ['1', '2', '3', ..., '9']

print(f"Checking file counts in subdirectories of '{os.path.abspath(parent_dir)}'...\n")

# Loop through each directory name
for name in dir_names:
    # Create the full path to the directory
    path = os.path.join(parent_dir, name)

    try:
        # Check if the path exists and is a directory
        if os.path.isdir(path):
            # Get a list of all entries in the directory
            all_entries = os.listdir(path)
            
            # Count only the entries that are files
            file_count = sum(1 for entry in all_entries if os.path.isfile(os.path.join(path, entry)))
            
            print(f"Directory '{name}' contains {file_count} files.")
        else:
            # This handles the case where a path like '1' exists but is a file, not a folder.
            print(f"Error: '{name}' is not a directory.")

    except FileNotFoundError:
        # This handles the case where the directory doesn't exist at all.
        print(f"Error: Directory '{name}' not found.")
    except Exception as e:
        # Catch any other potential errors, like permission issues.
        print(f"An unexpected error occurred with directory '{name}': {e}")