#define BLUR_SIZE 5

#include <png.h>

int width, height;
png_byte color_type;
png_byte bit_depth;
png_bytep *row_pointers = NULL;

unsigned char* read_png_file(const char *filename) 
{
    FILE *fp = fopen(filename, "rb");
    png_byte bit_depth;
    png_byte color_type;
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info = png_create_info_struct(png);
    png_init_io(png, fp);
    png_read_info(png, info);

    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    color_type = png_get_color_type(png, info);
    bit_depth = png_get_bit_depth(png, info);

    if(color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);
    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);
    if(png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);
    if(color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    if(color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    int numchan = 4;

    // Set up row pointer
    png_bytep *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    unsigned int i, j;
    for (i = 0; i < height; i++)
        row_pointers[i] = (png_bytep)malloc(png_get_rowbytes(png,info));
    png_read_image(png, row_pointers);

    // Put row pointers data into image
    unsigned char *image = (unsigned char *) malloc (numchan*width*height);
    int count = 0;
    for (i = 0 ; i < height ; i++)
    {
        for (j = 0 ; j < numchan*width ; j++)
        {
            image[count] = row_pointers[i][j];
            count += 1;
        }
    }
    fclose(fp);
    for (i = 0; i < height; i++)
        free(row_pointers[i]) ;
    free(row_pointers) ;

    return image;     
}

void write_png_file(unsigned char *filename) {
    int y;
  
    FILE *fp = filename;
    if(!fp) abort();
  
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) abort();
  
    png_infop info = png_create_info_struct(png);
    if (!info) abort();
  
    if (setjmp(png_jmpbuf(png))) abort();
  
    png_init_io(png, fp);
  
    // Output is 8bit depth, RGBA format.
    png_set_IHDR(
      png,
      info,
      width, height,
      8,
      PNG_COLOR_TYPE_RGBA,
      PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_DEFAULT,
      PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);
  
    // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
    // Use png_set_filler().
    //png_set_filler(png, 0, PNG_FILLER_AFTER);
  
    if (!row_pointers) abort();
  
    png_write_image(png, row_pointers);
    png_write_end(png, NULL);
  
    for(int y = 0; y < height; y++) {
      free(row_pointers[y]);
    }
    free(row_pointers);
  
    fclose(fp);
  
    png_destroy_write_struct(&png, &info);
  }

__global__ void blurKernel(unsigned char*in, unsigned char*out, int w, int h)
{
    int Col = blockIdx.x*blockDim.x + threadIdx.x;
    int Row = blockIdx.y*blockDim.y + threadIdx.y;

    if (Col < w && Row < h)
    {
        int pixR = 0; int pixG = 0; int pixB = 0;
        int pixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; blurRow++)
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; blurCol++)
            {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;

                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w)
                {
                    pixR += in[curRow*w*3 + curCol*3 + 0];
                    pixG += in[curRow*w*3 + curCol*3 + 1];
                    pixB += in[curRow*w*3 + curCol*3 + 2];
                    pixels++;
                }
            }
            out[3*(Row*w + Col)    ] = (unsigned char)(pixR / pixels);
            out[3*(Row*w + Col) + 1] = (unsigned char)(pixG / pixels);
            out[3*(Row*w + Col) + 2] = (unsigned char)(pixB / pixels);
    }
}

void process_png_file(unsigned char *image) {
    for(int y = 0; y < height; y++) {
      png_bytep row = row_pointers[y];
      for(int x = 0; x < width; x++) {
        png_bytep px = &(row[x * 4]);
        blurKernel <<<image, NULL>>> (px, px, width, height);
      }
    }
  }

int main(int ac, char **av)
{
    unsigned char *image = read_png_file(av[1]);
    process_png_file(image);
    write_png_file("out.png");
    return (0);
}