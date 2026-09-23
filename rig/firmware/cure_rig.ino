// Thermistor + heater rig for the composite cure demo.
// Streams temperature (Celsius) over serial, one reading per line.

const int THERMISTOR_PIN = A0;
const int HEATER_PIN = 9;

float readTemperatureC() {
  int raw = analogRead(THERMISTOR_PIN);
  float voltage = raw * (5.0 / 1023.0);
  float resistance = (5.0 - voltage) * 10000.0 / voltage; // 10k pull-up
  float steinhart = log(resistance / 10000.0) / 3950.0 + 1.0 / (25.0 + 273.15);
  return (1.0 / steinhart) - 273.15;
}

void setup() {
  Serial.begin(9600);
  pinMode(HEATER_PIN, OUTPUT);
}

void loop() {
  Serial.println(readTemperatureC(), 2);
  delay(1000);
}
