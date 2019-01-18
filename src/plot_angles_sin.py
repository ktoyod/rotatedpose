import sys

from visualize import save_angles_sin_plot


def main():
    args = sys.argv

    ANGLES_FILE_PATH = args[1]
    OUTPUT_PATH = args[2]

    save_angles_sin_plot(ANGLES_FILE_PATH, OUTPUT_PATH)


if __name__ == '__main__':
    main()
