#include <allegro.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
using namespace std;

const int bScreen = 200;
const int hScreen = 200;
BITMAP *buffer;

struct Cell {
  int i;
  int j;
};
const int neighborhood_size=9;
Cell V[neighborhood_size];

void getNeighborhoodV(int i, int j)
{
  V[0].i = i;
  V[0].j = j;

  V[1].i = i - 1;
  V[1].j = j;

  V[2].i = i;
  V[2].j = j - 1;

  V[3].i = i;
  V[3].j = j + 1;

  V[4].i = i + 1;
  V[4].j = j;

  V[5].i = i - 1;
  V[5].j = j - 1;

  V[6].i = i + 1;
  V[6].j = j - 1;

  V[7].i = i + 1;
  V[7].j = j + 1;

  V[8].i = i - 1;
  V[8].j = j + 1;
}

int white; // = makecol(255,255,255);
int black; //= makecol( 0, 255, 128);
int steps = 0;   // steps

const int d = bScreen;
double minimum = -22;
double maximum = 22;
int zoom = 2;

double curr_buff[d][d];
double next_buff[d][d];

void update() {
  for (int y = 0; y < bScreen; y++)
    for (int x = 0; x < hScreen; x++) {
      curr_buff[y][x] = next_buff[y][x];
    }
}

void transitionFunction(int x, int y) {
  getNeighborhoodV(y,x);
  if (y > 0 && x > 0 && curr_buff[y][x] != maximum) {
    double new_temp = 0;
    for (int n = 1; n < 5; ++n) {
      new_temp += curr_buff[V[n].i][V[n].j];
    }
    new_temp*=4.0;
    for (int n = 5; n < neighborhood_size; ++n) {
      new_temp += curr_buff[V[n].i][V[n].j];
    }
    new_temp/=20.0;

    next_buff[y][x] = new_temp;
  }
}

void globalTransictioFunction(){
  for (int y = 1; y < bScreen-1; y++)
    for (int x = 1; x < hScreen-1; x++) 
      transitionFunction(x, y);
  update();
  steps++;
}

void allegroInit(){
  allegro_init();
  install_keyboard();
  install_mouse();
  set_color_depth(24);
  set_gfx_mode( GFX_AUTODETECT_WINDOWED, bScreen*zoom, hScreen*zoom, 0, 0);
  show_mouse(screen);
  buffer = create_bitmap(bScreen*zoom, hScreen*zoom);
  white = makecol(255, 255, 255);
  black = makecol(0, 0, 0);
}

void heatTransferInit(){
  int firstLayers = 10;

  for (int y = firstLayers; y < bScreen-firstLayers; y++)
    for (int x = 0; x < hScreen; x++) {
      curr_buff[y][x] = minimum;
      next_buff[y][x] = minimum;
    }
  for (int y = 0; y < firstLayers; y++)
    for (int x = 0; x < hScreen; x++) {
      curr_buff[y][x] = maximum;
      next_buff[y][x] = maximum;
    }
  for (int y = bScreen-firstLayers; y < bScreen; y++)
    for (int x = 0; x < hScreen; x++) {
      curr_buff[y][x] = maximum;
      next_buff[y][x] = maximum;
    }
}

double max(double a, double b)
{
  if(a >= b)
    return a;
  return b;
}

void drawWithAllegro()
{
  //draw every pixel with a color depending on the state
  for (int y = 0; y < bScreen; y++)
    for (int x = 0; x < hScreen; x++)
    {
      int g = 0;
      int b = 0;
      int r = (curr_buff[x][y]-minimum)/(maximum - minimum)*255;

      int color = makecol(r,g,b);
      rectfill(buffer,y*zoom,x*zoom, (y+1)*zoom,(x+1)*zoom, color);
    }
  textprintf_ex(buffer, font, 0, 0, white, black, "Step: %i", steps);
  blit(buffer, screen, 0, 0, 0, 0, bScreen*zoom, hScreen*zoom);
}

int main(int, char**) {
  srand(time(NULL));
  allegroInit();
  heatTransferInit();

  bool pause_button_pressed= true;
  bool exitWhile = true;

  while (!key[KEY_ESC])
  {
    if(key[KEY_P])
      pause_button_pressed = true;

    if(key[KEY_R])
      pause_button_pressed = false;

    if(key[KEY_ESC])
      exitWhile = false;

    if (!pause_button_pressed)
      globalTransictioFunction();

    drawWithAllegro();
  }
  return 0;
}
END_OF_MAIN()
