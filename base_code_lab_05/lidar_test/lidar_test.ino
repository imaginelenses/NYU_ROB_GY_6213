// RPLIDAR A1 library test for Arduino Giga R1
// Uses a local copy of the RPLidar library (with the end() fix applied).
// All headers and RPLidar.cpp live in this sketch folder for Arduino IDE compatibility.

#include "RPLidar.h"

RPLidar lidar;


#define MOTOR_PIN 3

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("=== RPLIDAR A1 library test ===");

  pinMode(MOTOR_PIN, OUTPUT);
  analogWrite(MOTOR_PIN, 255);
  delay(2000); // wait for motor to reach speed
  Serial.println("Motor running.");

  // Start Serial2 ourselves first, then hand it to the library.
  // This avoids the library's begin() calling end() on an uninitialised port,
  // which hangs on the Giga R1's Mbed OS UART driver.
  Serial2.begin(115200);
  delay(100);
  lidar.begin(Serial2);
  delay(500);

  rplidar_response_device_info_t info;
  if (IS_OK(lidar.getDeviceInfo(info, 500))) {
    Serial.print("Firmware: ");
    Serial.print(info.firmware_version >> 8);
    Serial.print(".");
    Serial.println(info.firmware_version & 0xFF);
    Serial.print("Hardware: ");
    Serial.println(info.hardware_version);
  } else {
    Serial.println("ERROR: getDeviceInfo failed. Check wiring.");
  }

  if (IS_OK(lidar.startScan())) {
    Serial.println("Scan started.");
    Serial.println("angle(deg)\tdistance(mm)\tquality");
  } else {
    Serial.println("ERROR: startScan failed.");
  }
}

void loop() {
  if (IS_OK(lidar.waitPoint(100))) {
    float    distance = lidar.getCurrentPoint().distance;
    float    angle    = lidar.getCurrentPoint().angle;
    uint8_t  quality  = lidar.getCurrentPoint().quality;

    Serial.print(angle, 1);
    Serial.print("\t");
    Serial.print((int)distance);
    Serial.print(" mm\t");
    Serial.println(quality);
  }
}
