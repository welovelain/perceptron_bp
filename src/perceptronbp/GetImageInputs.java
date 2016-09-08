package perceptronbp;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class GetImageInputs {
    
    static final int BLACK = -16777216;
    static final int WHITE = -1;
    static final int GREEN = -14503604;
    
    // the squares are 36x36 pixels    
    static double[] get(String filepath) {
        
        double[] result = new double[24];
        
        try {
            BufferedImage image = ImageIO.read(new File(filepath));    
            
            int pixelColor = 0;
            int x = 5;
            int y = 5;
            
            int count = 0;
            for (int i = 0; i < 6; ++i ) {
                for (int j = 0; j < 4; ++j ) {
                    if (image.getRGB(x, y) == GREEN) result[count] = 1;
                    else result[count] = 0;
                    
                    if (j != 4) x+=35;
                    ++count;
                }
                x = 5; y+=36;
                
            }
            
        } 
        catch (IOException e)
        {
            e.printStackTrace();
        }
        
        
       return result;
    }
    
}
