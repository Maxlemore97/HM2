# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt


# Function to create 3D plots
def plot_3d(x, y, z, plot_type='wireframe', title='', xlabel='', ylabel='', zlabel='', cmap='viridis'):
  """
  Creates a 3D plot using matplotlib.

  Parameters:
  - x: 2D array-like, meshgrid of x-values (e.g., np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50)))
  - y: 2D array-like, meshgrid of y-values (e.g., same shape as x)
  - z: 2D array-like, meshgrid of z-values (e.g., function of x and y, same shape as x and y)
  - plot_type: str, type of 3D plot ('wireframe' or 'surface')
      - 'wireframe': Creates a wireframe plot
      - 'surface': Creates a surface plot with color mapping
  - title: str, title of the plot (e.g., "3D Plot Example")
  - xlabel: str, label for the x-axis (e.g., "X-Axis Label")
  - ylabel: str, label for the y-axis (e.g., "Y-Axis Label")
  - zlabel: str, label for the z-axis (e.g., "Z-Axis Label")
  - cmap: str, colormap for surface plots (e.g., 'viridis', 'plasma', 'coolwarm')

  Example:
  x, y = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
  z = np.sin(x) * np.cos(y)
  plot_3d(x, y, z, plot_type='surface', title="Example Surface Plot", xlabel="X", ylabel="Y", zlabel="Z", cmap='plasma')
  """
  fig = plt.figure()  # Create a new figure
  ax = fig.add_subplot(111, projection='3d')  # Add a 3D subplot

  if plot_type == 'wireframe':
    # Create a wireframe plot
    ax.plot_wireframe(x, y, z, color='blue')  # Wireframe color is blue
  elif plot_type == 'surface':
    # Create a surface plot
    surf = ax.plot_surface(x, y, z, cmap=cmap)  # Use the specified colormap
    fig.colorbar(surf)  # Add a colorbar to the figure

  # Set axis labels and title
  ax.set_xlabel(xlabel)  # Set x-axis label
  ax.set_ylabel(ylabel)  # Set y-axis label
  ax.set_zlabel(zlabel)  # Set z-axis label
  ax.set_title(title)  # Set plot title

  plt.show()  # Display the plot


# Function to create 2D contour plots
def plot_2d_contour(x, y, z, title='', xlabel='', ylabel='', cmap='viridis'):
  """
  Creates a 2D contour plot using matplotlib.

  Parameters:
  - x: 2D array-like, meshgrid of x-values (e.g., np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50)))
  - y: 2D array-like, meshgrid of y-values (e.g., same shape as x)
  - z: 2D array-like, meshgrid of z-values (e.g., function of x and y, same shape as x and y)
  - title: str, title of the plot (e.g., "2D Contour Plot Example")
  - xlabel: str, label for the x-axis (e.g., "X-Axis Label")
  - ylabel: str, label for the y-axis (e.g., "Y-Axis Label")
  - cmap: str, colormap for the contour plot (e.g., 'viridis', 'plasma', 'coolwarm')

  Example:
  x, y = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
  z = np.sin(x) * np.cos(y)
  plot_2d_contour(x, y, z, title="Example Contour Plot", xlabel="X", ylabel="Y", cmap='plasma')
  """
  fig, ax = plt.subplots()  # Create a new figure and axis
  c = ax.contourf(x, y, z, cmap=cmap)  # Create a filled contour plot with the specified colormap
  fig.colorbar(c)  # Add a colorbar to the figure

  # Set axis labels and title
  ax.set_xlabel(xlabel)  # Set x-axis label
  ax.set_ylabel(ylabel)  # Set y-axis label
  ax.set_title(title)  # Set plot title

  plt.show()  # Display the plot


def main():
    # Example data for 3D plot
    x, y = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
    z = np.sin(x) * np.cos(y)

    # Call the 3D plot function
    plot_3d(
        x, y, z,
        plot_type='surface',
        title="3D Surface Plot Example",
        xlabel="X-Axis",
        ylabel="Y-Axis",
        zlabel="Z-Axis",
        cmap='plasma'
    )

    # Example data for 2D contour plot
    z_contour = np.sin(x) + np.cos(y)

    # Call the 2D contour plot function
    plot_2d_contour(
        x, y, z_contour,
        title="2D Contour Plot Example",
        xlabel="X-Axis",
        ylabel="Y-Axis",
        cmap='viridis'
    )

# Run the main function
if __name__ == "__main__":
    main()