.. _topgear_dataset:

TopGear dataset
--------------------

**Data Set Characteristics:**

- Number of Instances: 297 
- Number of Attributes: 32 (13 numeric, 19 categorical)
- Attribute Information:
    * Maker (str): the car maker.
    * Model (str): the car model.
    * Type (str): the exact model type.
    * Fuel (str): the type of fuel ("Diesel" or "Petrol").
    * Price (float): the list price (in UK pounds)
    * Cylinders (float): the number of cylinders in the engine.
    * Displacement (float): the displacement of the engine (in cc).
    * DriveWheel (str): the type of drive wheel ("4WD", "Front" or "Rear").
    * BHP (float): the power of the engine (in bhp).
    * Torque (float): the torque of the engine (in lb/ft).
    * Acceleration (float): the time it takes the car to get from 0 to 62 mph (in seconds).
    * TopSpeed (float): the car's top speed (in mph).
    * MPG (float): the combined fuel consuption (urban + extra urban; in miles per gallon).
    * Weight (float): the car's curb weight (in kg).
    * Length (float): the car's length (in mm).
    * Width (float): the car's width (in mm).
    * Height (float): the car's height (in mm).
    * AdaptiveHeadlights (str): whether the car has adaptive headlights ("no", "optional" or "standard").
    * AdjustableSteering (str): whether the car has adjustable steering ("no" or "standard").
    * AlarmSystem (str) whether the car has an alarm system ("no/optional" or "standard").
    * Automatic (str) whether the car has an automatic transmission ("no", "optional" or "standard").
    * Bluetooth (str) whether the car has bluetooth ("no", "optional" or "standard").
    * ClimateControl (str) whether the car has climate control ("no", "optional" or "standard").
    * CruiseControl (str) whether the car has cruise control ("no", "optional" or "standard").
    * ElectricSeats (str) whether the car has electric seats ("no", "optional" or "standard").
    * Leather (str) whether the car has a leather interior ("no", "optional" or "standard").
    * ParkingSensors (str) whether the car has parking sensors ("no", "optional" or "standard").
    * PowerSteering (str) whether the car has power steering ("no" or "standard").
    * SatNav (str) whether the car has a satellite navigation system ("no", "optional" or "standard").
    * ESP (str) whether the car has ESP ("no", "optional" or "standard").
    * Verdict (float) review score between 1 (lowest) and 10 (highest).
    * Origin (str) the origin of the car maker ("Asia", "Europe" or "USA").

- Creator: BBC TopGear
- Source: https://rdrr.io/cran/robustHD/man/TopGear.html

**Data Set Description:**
The data set contains information on cars featured on the website of the popular BBC television show Top Gear.
The data were scraped from http://www.topgear.com/uk/ on 2014-02-24. Variable Origin was added based on the car maker information.


