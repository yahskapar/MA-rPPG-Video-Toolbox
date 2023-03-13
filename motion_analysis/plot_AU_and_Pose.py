import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the 17 facial action units (AUs) of interest
aus_of_interest = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]

# Define the 3 pose rotations of interest
pose_rotations_of_interest = ['pose_Rx', 'pose_Ry', 'pose_Rz']

# Define the directory where the OpenFace CSV files are located
input_dir = '/path/to/input/dir'

# Initialize empty lists to store the mean standard deviation values
au_intensity_std_means = []
pose_rotation_std_means = []

# Loop over all CSV files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        # if 'hp' not in filename:
        #     continue
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(os.path.join(input_dir, filename))
        
        # Extract the pose rotation data from the DataFrame
        pose_rotations = df[['frame', 'timestamp'] + pose_rotations_of_interest]

        # Compute the standard deviation of each pose rotation
        pose_rotation_std = pose_rotations[[col for col in pose_rotations.columns if col not in ['frame', 'timestamp']]].std()

        # Compute the mean of the standard deviations of all pose rotations
        pose_rotation_std_mean = pose_rotation_std.mean()
        
        # Append the mean standard deviation value to the list
        pose_rotation_std_means.append(pose_rotation_std_mean)

        # Extract the AU intensity data from the DataFrame
        aus_intensity = df[['frame', 'timestamp'] + ['AU{:02d}_r'.format(au) for au in aus_of_interest]]

        # Compute the standard deviation of each AU intensity
        au_intensity_std = aus_intensity[[col for col in aus_intensity.columns if col not in ['frame', 'timestamp']]].std()

        # Compute the mean of the standard deviations of all AUs
        au_intensity_std_mean = au_intensity_std.mean()
        
        # Append the mean standard deviation value to the list
        # au_intensity_std_means.append(au_intensity_std_mean/pose_rotation_std_mean)
        au_intensity_std_means.append(au_intensity_std_mean)

# Calculate Overall Means and Medians
au_mean = np.mean(au_intensity_std_means)
au_median = np.median(au_intensity_std_means)
pose_mean = np.mean(pose_rotation_std_means)
pose_median = np.median(pose_rotation_std_means)

# Plot the mean standard deviation values for the AU intensities and pose rotations as subplots and save the figure
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].hist(au_intensity_std_means)
axs[0].set_xlabel('Mean standard deviation (AU intensity)')
axs[0].set_ylabel('Frequency')
axs[0].set_title(f'Mean:{au_mean:.3f}, Median:{au_median:.3f}')
axs[1].hist(pose_rotation_std_means)
axs[1].set_xlabel('Mean standard deviation (pose rotations)')
axs[1].set_ylabel('Frequency')
axs[1].set_title(f'Mean:{pose_mean:.3f}, Median:{pose_median:.3f}')
fig.tight_layout()
plt.savefig('filter_au_pose_mean_std_subplots.png')
