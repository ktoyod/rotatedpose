import numpy as np
from PIL import Image, ImageDraw


# draw the lines
def drawLines(draw, joints, clr, line_width):
    draw.line((joints[1, 0], joints[1, 1], joints[2, 0], joints[2, 1]), fill=(
        clr[0, 0], clr[0, 1], clr[0, 2]), width=line_width)
    draw.line((joints[1, 0], joints[1, 1], joints[5, 0], joints[5, 1]), fill=(
        clr[1, 0], clr[1, 1], clr[1, 2]), width=line_width)
    draw.line((joints[2, 0], joints[2, 1], joints[3, 0], joints[3, 1]), fill=(
        clr[2, 0], clr[2, 1], clr[2, 2]), width=line_width)
    draw.line((joints[3, 0], joints[3, 1], joints[4, 0], joints[4, 1]), fill=(
        clr[3, 0], clr[3, 1], clr[3, 2]), width=line_width)
    draw.line((joints[5, 0], joints[5, 1], joints[6, 0], joints[6, 1]), fill=(
        clr[4, 0], clr[4, 1], clr[4, 2]), width=line_width)
    draw.line((joints[6, 0], joints[6, 1], joints[7, 0], joints[7, 1]), fill=(
        clr[5, 0], clr[5, 1], clr[5, 2]), width=line_width)
    draw.line((joints[1, 0], joints[1, 1], joints[8, 0], joints[8, 1]), fill=(
        clr[6, 0], clr[6, 1], clr[6, 2]), width=line_width)
    draw.line((joints[8, 0], joints[8, 1], joints[9, 0], joints[9, 1]), fill=(
        clr[7, 0], clr[7, 1], clr[7, 2]), width=line_width)
    draw.line((joints[9, 0], joints[9, 1], joints[10, 0], joints[10, 1]), fill=(
        clr[8, 0], clr[8, 1], clr[8, 2]), width=line_width)
    draw.line((joints[1, 0], joints[1, 1], joints[11, 0], joints[11, 1]), fill=(
        clr[9, 0], clr[9, 1], clr[9, 2]), width=line_width)
    draw.line((joints[11, 0], joints[11, 1], joints[12, 0], joints[12, 1]), fill=(
        clr[10, 0], clr[10, 1], clr[10, 2]), width=line_width)
    draw.line((joints[12, 0], joints[12, 1], joints[13, 0], joints[13, 1]), fill=(
        clr[11, 0], clr[11, 1], clr[11, 2]), width=line_width)
    draw.line((joints[0, 0], joints[0, 1], joints[1, 0], joints[1, 1]), fill=(
        clr[12, 0], clr[12, 1], clr[12, 2]), width=line_width)
    draw.line((joints[0, 0], joints[0, 1], joints[14, 0], joints[14, 1]), fill=(
        clr[13, 0], clr[13, 1], clr[13, 2]), width=line_width)
    draw.line((joints[15, 0], joints[15, 1], joints[17, 0], joints[17, 1]), fill=(
        clr[14, 0], clr[14, 1], clr[14, 2]), width=line_width)
    draw.line((joints[0, 0], joints[0, 1], joints[15, 0], joints[15, 1]), fill=(
        clr[15, 0], clr[15, 1], clr[15, 2]), width=line_width)
    draw.line((joints[0, 0], joints[0, 1], joints[15, 0], joints[15, 1]), fill=(
        clr[16, 0], clr[16, 1], clr[16, 2]), width=line_width)


# draw the elements transparent
def draw_transparent(canvas, alpha, joints, clr, joint_size, line_width):
    mask = Image.new("L", canvas.size, 1)
    a_canvas = Image.new("RGB", canvas.size,  (255, 255, 255))
    a_canvas.putalpha(mask)
    draw = ImageDraw.Draw(a_canvas)

    drawLines(draw, joints, clr, line_width)

    for i in range(18):
        draw.ellipse((joints[i, 0]-joint_size, joints[i, 1]-joint_size,
                      joints[i, 0]+joint_size, joints[i, 1]+joint_size),
                     fill=(clr[i, 0], clr[i, 1], clr[i, 2]), outline=None)

    del draw

    canvas.putalpha(mask)
    return Image.blend(canvas, a_canvas, alpha).convert("RGB")


def draw_joints(img_path, output_path, reconst_joints):
    # define color params
    c1 = 255
    c2 = c1-(int(c1/4))
    c3 = c2-(int(c1/4))
    c4 = 0

    # define the colors
    set_clr = np.array([[c1, c4, c4],
                        [c1, c3, c4],
                        [c1, c2, c4],
                        [c1, c1, c4],
                        [c2, c1, c4],
                        [c3, c1, c4],
                        [c4, c1, c4],
                        [c4, c1, c3],
                        [c4, c1, c2],
                        [c4, c1, c1],
                        [c4, c2, c1],
                        [c4, c3, c1],
                        [c4, c4, c1],
                        [c3, c4, c1],
                        [c2, c4, c1],
                        [c1, c4, c2],
                        [c1, c4, c1],
                        [c1, c4, c3]])

    # set size of joints
    joint_size = 5

    # set line width
    line_width = 7

    # set alpha
    alpha_val = 0.5

    # read image here
    canvas = Image.open(img_path)

    canvas = draw_transparent(
        canvas, alpha_val, reconst_joints, set_clr, joint_size, line_width)

    # save the images
    canvas.save(output_path)
