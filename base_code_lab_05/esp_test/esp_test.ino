// ESP32 AT Command Test Sketch for Arduino GIGA R1 WiFi
// HardwareSerial: Serial3 (TX2/RX2) on GIGA
// Baud: 115200 (default for ESP-AT firmware)
//
// Wire ESP32 TX2 (GPIO17) -> GIGA pin 17 (RX2 = Serial3 RX)
//      ESP32 RX2 (GPIO16) -> GIGA pin 16 (TX2 = Serial3 TX)
//      ESP32 GND -> GIGA GND
//
// Open Serial Monitor at 115200 baud. Type AT commands to interact with ESP32.

void setup() {
  Serial.begin(115200);    // USB serial for monitor
  Serial3.begin(115200);   // ESP32 AT UART
  delay(2000);             // Wait for ESP32 to boot
  Serial3.println("AT");   // Send AT command
  Serial.println("Sent AT");
}

void loop() {
  // Forward ESP32 responses to Serial Monitor
  while (Serial3.available()) Serial.write(Serial3.read());
  // Forward user input to ESP32
  while (Serial.available()) Serial3.write(Serial.read());
}
