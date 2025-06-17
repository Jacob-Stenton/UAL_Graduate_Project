
#define upButton 10
#define downButton 11
#define enterButton 12

int upButtonState = 0;
int downButtonState = 0;
int enterButtonState = 0;

int lastUpButtonState = 0;
int lastDownButtonState = 0;
int lastEnterButtonState = 0;

void setup() {
  Serial.begin(9600);
  pinMode(upButton, INPUT_PULLUP);
  pinMode(downButton, INPUT_PULLUP);
  pinMode(enterButton, INPUT_PULLUP);
}

void loop() {
  upButtonState = digitalRead(upButton);
  downButtonState = digitalRead(downButton);
  enterButtonState = digitalRead(enterButton);

  if (upButtonState == LOW && lastUpButtonState == HIGH) {
    Serial.println("up");
    delay(50); 
  }

  if (downButtonState == LOW && lastDownButtonState == HIGH) {
    Serial.println("down");
    delay(50); 
  }

  if (enterButtonState == LOW && lastEnterButtonState == HIGH) {
    Serial.println("enter");
    delay(50); 
  }

  lastUpButtonState = upButtonState;
  lastDownButtonState = downButtonState;
  lastEnterButtonState = enterButtonState;
}