# AI-Based Driver Drowsiness Detection System

An AI-based driver drowsiness detection system that monitors the driver's eye state in real time and provides alerts when prolonged eye closure is detected.

## Features

* Real-time driver face and eye monitoring
* Eye Aspect Ratio (EAR) based drowsiness detection
* Audible warning alarm
* Seat vibration alert
* Simulated automatic braking
* GPS location detection
* Emergency notification through Telegram
* Emergency notification through Twilio SMS

## Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy
* SciPy
* Pygame
* Telegram Bot API
* Twilio API
* Geocoder

## Project Versions

### Telegram Version

Sends an emergency alert with the detected location through a Telegram bot when the vehicle is stopped after prolonged driver drowsiness.

### Twilio Version

Sends an emergency SMS containing the detected location using the Twilio API.

## How It Works

1. Start the system by pressing `S`.
2. The camera continuously monitors the driver's eyes.
3. Eye closure is detected using Eye Aspect Ratio (EAR).
4. If the eyes remain closed:

   * An audible alarm is activated.
   * Seat vibration is activated.
   * Simulated automatic braking begins.
5. When the simulated vehicle speed reaches zero, an emergency location alert is sent through Telegram or Twilio.
6. The system displays the driver's status, EAR value, and simulated vehicle speed on the screen.

## Planned Enhancements

The following features are planned for the hardware-integrated version:

* Physical buzzer activation during vehicle deceleration and braking
* Blinking warning lights to alert vehicles behind during braking
* Integration with ESP32 and external vehicle warning hardware
* Real-time vehicle speed and braking control using hardware sensors

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Configuration

API credentials are stored locally in a `.env` file and are excluded from GitHub using `.gitignore`.

**Never upload API keys, bot tokens, passwords, or other sensitive credentials to GitHub.**

## Note

This project is a prototype/research implementation. The vehicle speed, automatic braking, buzzer, and warning lights are simulated or planned in software and are not connected to an actual vehicle control system.
