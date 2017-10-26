package perceptronbp.input;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ImageInputExtractor {
    private final int PIXEL_FILLED = 1;
    private final int PIXEL_EMPTY = 0;

    // Code of filled square. For example, green = -14503604;
    private int filledColorCode; // -14503604

    // width and height of single "pixel".
    private int pixelSize; // 36

    // amount of pixels in height
    private int height; // 6

    // amount of pixels in width.
    private int width; // 4

    // border of pixel if any
    private int border; // 1

    public ImageInputExtractor(int filledColorCode, int pixelSize, int height, int width) {
       this(filledColorCode, pixelSize, height, width, 0);
    }

    public ImageInputExtractor(int filledColorCode, int pixelSize, int height, int width, int border) {
        if (border >= height || border >= width) {
            throw new IllegalArgumentException("Border should be < than width or height");
        }
        if (border < 0 ) {
            throw new IllegalArgumentException("Border should be >= 0");
        }
        if (pixelSize <= 0) {
            throw new IllegalArgumentException("Pixel size should be > 0");
        }
        if (height <= 0) {
            throw new IllegalArgumentException("Height size should be > 0");
        }
        if (width <= 0) {
            throw new IllegalArgumentException("Width size should be > 0");
        }

        this.filledColorCode = filledColorCode;
        this.pixelSize = pixelSize;
        this.width = width;
        this.height = height;
        this.border = border;
    }

    public double[] get(String filepath) throws IOException {
        double[] result = new double[height * width];
        BufferedImage pixel = getImageFromFile(filepath);

        // ignore borders
        int x = border + 1;
        int y = border + 1;

        int count = 0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                result[count++] = (pixel.getRGB(x, y) == filledColorCode) ? PIXEL_FILLED : PIXEL_EMPTY;
                x += pixelSize;
            }
            x = border + 1;
            y += pixelSize;
        }

        return result;
    }

    private BufferedImage getImageFromFile(String filepath) throws IOException {
        ClassLoader cl = getClass().getClassLoader();
        return ImageIO.read(new File(cl.getResource(filepath).getFile()));
    }

//    // the squares are 36x36 pixels
//    static double[] get(String filepath) throws IOException {
//
//        double[] result = new double[height * width];
//
//
//        BufferedImage image = getImageFromFile(filepath);
//
//        int pixelColor = 0;
//        int x = 5;
//        int y = 5;
//
//        int count = 0;
//        for (int i = 0; i < 6; ++i ) {
//            for (int j = 0; j < 4; ++j ) {
//                if (image.getRGB(x, y) == GREEN) result[count] = 1;
//                else result[count] = 0;
//
//                if (j != 4) x+=35;
//                ++count;
//            }
//            x = 5; y+=36;
//
//        }
//
//
//
//
//        return result;
//    }



}
