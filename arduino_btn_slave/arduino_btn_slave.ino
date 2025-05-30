#include <Wire.h>

#define upButton 10
#define downButton 11
#define enterButton 12

int upButtonVal = 0;
int downButtonVal = 0;
int enterButtonVal = 0;

void setup() {
  Wire.begin(0x8);

  pinMode(upButton, INPUT_PULLUP);
  pinMode(downButton, INPUT_PULLUP);
  pinMode(enterButton, INPUT_PULLUP);

}

void loop() {
  
  upButtonVal = digitalRead(upButton);
  downButtonVal = digitalRead(downButton);
  enterButtonVal = digitalRead(enterButton);

  if (upButtonVal == LOW) {
    Wire.write("up");
  }

  if (downButtonVal == LOW) {
    Wire.write("down");
  }

  if (enterButtonVal == LOW) {
    Wire.write("enter");
  }

  delay(100);
}
