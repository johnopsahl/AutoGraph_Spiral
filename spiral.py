import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import svgwrite
from PIL import Image

def calc_spiral_arc_length(theta, b):
    """Calculates arc length L of spiral r = a * theta from 0 to theta."""
    if theta == 0:
        return 0.0
    # Exact formula for Archimedean spiral arc length
    return (b / 2.0) * (theta * np.sqrt(1 + theta**2) + np.log(theta + np.sqrt(1 + theta**2)))

def find_theta_from_length(target_length, b, initial_theta_guess):
    """Numerically solves for theta given a target arc length."""
    # Define the function whose root we want to find: f(theta) - target_L = 0
    func = lambda t: calc_spiral_arc_length(t, b) - target_length

    # root() expects the function to accept and return array-like values
    result = root(func, initial_theta_guess)

    if not result.success:
        raise RuntimeError(f"Root finding failed: {result.message}")

    return result.x[0]

def generate_spiral_segment_points(a, b, segment_length, theta_start, theta_end):
    """Generates line segment points along an Archimedes spiral."""

    # Create an array of theta values from 0 to theta_end with a step size determined by the desired segment length
    theta = []
    theta_current = theta_start

    while theta_current < theta_end:
        length_initial = calc_spiral_arc_length(theta_current, b)
        length_target = length_initial + segment_length
        theta_new = find_theta_from_length(length_target, b, theta_current)
        theta.append(theta_new)
        theta_current = theta_new
    theta = np.array(theta)

    # Calculate the radius for each theta using the formula r = a + b*theta
    r = a + b * theta
    
    # Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return np.column_stack((x,y))

def create_segments_from_segment_points(seg_points):

    return np.lib.stride_tricks.sliding_window_view(seg_points, 2, axis=0).transpose(0, 2, 1)

def remove_out_of_bounds_segments(segment, x_min, x_max, y_min, y_max):
    x_coords = segment[:, :, 0]  # shape (N, 2)
    y_coords = segment[:, :, 1]  # shape (N, 2)

    mask = (
        (x_coords >= x_min).all(axis=1) &
        (x_coords <= x_max).all(axis=1) &
        (y_coords >= y_min).all(axis=1) &
        (y_coords <= y_max).all(axis=1)
    )

    return segment[mask]

def create_segment_centers(segment):

    return np.mean(segment, axis=(1))

def write_segments_to_gcode(gcode_filename, segment, segment_color, feedrate):

    with open(gcode_filename + '.gcode', 'w') as f:
        for i, seg in enumerate(segment):
            
            if i == 0:
                # move to starting point of first segment and set feedrate for whole drawing
                f.write(f"G0 X{seg[0][0]:.3f} Y{seg[0][1]:.3f} F{feedrate:.2f}\n")
                f.write("M3 S0\n")  # turn on pwm at 0  
            else:
                # G0 move if beginning of segment is not the same as end of previous segment
                if not np.array_equal(segment[i-1][1], segment[i][0]):
                    f.write(f"G0 X{seg[1][0]:.3f} Y{seg[1][1]:.3f} S0\n")

            # convert color to grayscale value between 0 and 255
            color_value = segment_color[i]
            # interpolate grayscale value to pencil force
            pencil_force = np.interp(color_value, [0, 255], [1000, 0])

            f.write(f"G1 X{seg[1][0]:.3f} Y{seg[1][1]:.3f} S{pencil_force:.0f}\n")

    f.close()

def write_segments_to_svg(filename: str, segment, view_box):
    
    dwg = svgwrite.Drawing(filename + '.svg')

    for seg in segment:          

        dwg.add(dwg.line(seg[0], seg[1], stroke=svgwrite.rgb(0, 0, 0), stroke_width=0.1))

    dwg.viewbox(minx=view_box[0], miny=view_box[1], 
                width=view_box[2], height=view_box[3])
    dwg.save()

def write_segments_and_points_to_svg(filename, segment, seg_center, view_box):
    
    dwg = svgwrite.Drawing(filename + '.svg')

    for seg in segment:          

        dwg.add(dwg.line(seg[0], seg[1], stroke=svgwrite.rgb(0, 0, 0), stroke_width=0.1))

    for pnt in seg_center:

        dwg.add(dwg.circle(center=(pnt[0], pnt[1]), r=0.1, fill='blue', stroke='blue', stroke_width=0.1))
        
    dwg.viewbox(minx=view_box[0], miny=view_box[1], 
                width=view_box[2], height=view_box[3])
    dwg.save()

def write_segments_seg_centers_and_seg_center_pixels_to_svg(filename, segment, segement_color,
                                                            seg_center, seg_center_pixel, pixel_center_coord, view_box):
    
    dwg = svgwrite.Drawing(filename + '.svg')

    for i, seg in enumerate(segment):          

        dwg.add(dwg.line(seg[0], seg[1], stroke=svgwrite.rgb(segement_color[i], segement_color[i], segement_color[i]), stroke_width=0.1))

    # for pnt in seg_center:

    #     dwg.add(dwg.circle(center=(pnt[0], pnt[1]), r=0.1, fill='blue', stroke='blue', stroke_width=0.1))
    
    # for pnt in seg_center_pixel:

    #     dwg.add(dwg.circle(center=(pnt[0], pnt[1]), r=0.09, fill='red', stroke='red', stroke_width=0.09))

    # for pnt in pixel_center_coord:

    #     dwg.add(dwg.circle(center=(pnt[0], pnt[1]), r=0.05, fill='green', stroke='green', stroke_width=0.05))

    # dwg.add(dwg.circle(center=(0, 0), r=0.1, fill='purple', stroke='purple', stroke_width=0.1))

    dwg.viewbox(minx=view_box[0], miny=view_box[1], 
                width=view_box[2], height=view_box[3])
    dwg.save()

def display_spiral_segments(a, b, segment_length, theta_end):
    
    seg_point = generate_spiral_segment_points(a, b, segment_length, 0, theta_end)
    
    segment = create_segments_from_segment_points(seg_point)

    view_box = [-10, -10, 20, 20]
    write_segments_to_svg("spiral_segments", segment, view_box)

def clip_spiral_segments_to_image_boundaries(a, b, segment_length, theta_end):
    
    seg_point = generate_spiral_segment_points(a, b, segment_length, 0, theta_end)
    
    segment = create_segments_from_segment_points(seg_point)

    segment_filtered = remove_out_of_bounds_segments(segment, -5, 5, -5, 5)

    view_box = [-10, -10, 20, 20]
    write_segments_to_svg("spiral_segments_clipped", segment_filtered, view_box)

def display_spiral_segments_and_centers(a, b, segment_length, theta_end):
    
    seg_point = generate_spiral_segment_points(a, b, segment_length, 0, theta_end)

    segment = create_segments_from_segment_points(seg_point)

    segment_filtered = remove_out_of_bounds_segments(segment, -5, 5, -5, 5)

    #create segment centers from segments
    seg_center = create_segment_centers(segment_filtered)

    view_box = [-10, -10, 20, 20]
    write_segments_and_points_to_svg("spiral_segments_centers", 
                                     segment_filtered, seg_center, view_box)

def size_and_display_spiral_segments_and_centers(image_filename, drawing_width_mm, 
                                                 a, spiral_pitch_mm, segment_length_mm):
    
    img = Image.open(image_filename)
    image_pixel_width, image_pixel_height = img.size

    mm_per_pixel = drawing_width_mm/image_pixel_width
    drawing_height_mm = image_pixel_height*mm_per_pixel
    drawing_hypotenuse_mm = np.sqrt(drawing_width_mm**2 + drawing_height_mm**2)

    # calculate b based on desired spiral pitch
    b = spiral_pitch_mm/(2*np.pi)

    # using the drawing hypotenuse to determine the number of rotations is a bit hacky for all spiral center positions, 
    # but it ensures the spiral extends beyond the corners of the drawing
    num_rotations = drawing_hypotenuse_mm/(4*np.pi*b) + 1
    theta_end = num_rotations*2*np.pi

    seg_point = generate_spiral_segment_points(a, b, segment_length_mm, 0, theta_end)

    # translate center of spiral
    spiral_center = np.array([drawing_width_mm/2, drawing_height_mm/2])
    seg_point_trans = seg_point + spiral_center

    segment = create_segments_from_segment_points(seg_point_trans)

    segment_filtered = remove_out_of_bounds_segments(segment, 0, drawing_width_mm, 0, drawing_height_mm)

    # create segment centers from segments
    seg_center = create_segment_centers(segment_filtered)

    # assign pixel to each segment center, making sure to snap to pixel centers at 0.5 increments
    seg_center_pixel_index = np.floor(seg_center/mm_per_pixel).astype(int)
    seg_center_pixel_coords = seg_center_pixel_index + np.array([0.5, 0.5])
    seg_center_pixel_mm = seg_center_pixel_coords*mm_per_pixel 

    # convert image to grayscale
    img_gray = np.array(img.convert("L")) #convert image to grayscale

    # get color of each pixel index
    segment_color = img_gray[seg_center_pixel_index[:, 1], seg_center_pixel_index[:, 0]]

    # For troubleshooting
    # create a list of all pixel center coordinates in mm to compare against segment center pixels for debugging
    rows, cols = np.mgrid[0:image_pixel_width, 0:image_pixel_height]
    pixel_index = np.c_[rows.ravel(), cols.ravel()]
    pixel_center_coord = pixel_index + np.array([0.5, 0.5])
    pixel_center_coord_mm = pixel_center_coord*mm_per_pixel

    view_box = [0, 0, drawing_width_mm, drawing_height_mm]
    write_segments_seg_centers_and_seg_center_pixels_to_svg("spiral_sized", segment_filtered, segment_color, seg_center, 
                                                            seg_center_pixel_mm, pixel_center_coord_mm, view_box)
    
    # invert y axis for gcode since svg and image coordinates have y increasing downwards but gcode has y increasing upwards
    segment_filtered[:, :, 1] = drawing_height_mm - segment_filtered[:, :, 1]
    
    write_segments_to_gcode("gcode_output", segment_filtered, segment_color, 250)
    

if __name__ == '__main__':

    # convert_bitmap_to_spiral("margaret_gym.png", 0, 1.5, 5, theta_end)
    display_spiral_segments(0, 0.3, 0.5, 8*np.pi)
    clip_spiral_segments_to_image_boundaries(0, 0.3, 0.5, 8*np.pi)
    display_spiral_segments_and_centers(0, 0.05, 0.5, 64*np.pi)
    size_and_display_spiral_segments_and_centers(image_filename='margaret gym_gray.png', 
                                                 drawing_width_mm=100, 
                                                 a=0, 
                                                 spiral_pitch_mm=0.7, 
                                                 segment_length_mm=.25)
