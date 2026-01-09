#!/usr/bin/env python3
"""
Radio Telescope Dish Position Monitor
Tracks elevation (pitch) and azimuth (yaw) angles
Detects unwanted movement (wind, etc.)
"""

import asyncio
import struct
import time
from bleak import BleakClient

# Your sensor's MAC address
SENSOR_MAC = "E0:D6:FC:57:08:EF"

# Movement detection threshold (degrees per second)
MOVEMENT_THRESHOLD = 5.0  # Adjust this based on your needs

class DishMonitor:
    def __init__(self, address):
        self.address = address
        self.client = None
        self.notify_char = "0000ffe4-0000-1000-8000-00805f9a34fb"
        self.write_char = "0000ffe9-0000-1000-8000-00805f9a34fb"
        
        # Tracking
        self.elevation = 0.0  # Pitch angle
        self.azimuth = 0.0    # Yaw angle
        self.last_update = time.time()
        self.movement_detected = False
        
    def parse_data(self, data):
        """Parse the sensor data packet (20 bytes)"""
        if len(data) != 20 or data[0] != 0x55 or data[1] != 0x61:
            return None
        
        # Parse angular velocity (bytes 8-13) - for movement detection
        wx = struct.unpack('<h', data[8:10])[0] / 32768.0 * 2000  # °/s
        wy = struct.unpack('<h', data[10:12])[0] / 32768.0 * 2000
        wz = struct.unpack('<h', data[12:14])[0] / 32768.0 * 2000
        
        # Parse angles (bytes 14-19)
        roll = struct.unpack('<h', data[14:16])[0] / 32768.0 * 180  # degrees
        pitch = struct.unpack('<h', data[16:18])[0] / 32768.0 * 180  # ELEVATION
        yaw = struct.unpack('<h', data[18:20])[0] / 32768.0 * 180    # AZIMUTH
        
        return {
            'elevation': pitch,  # Vertical angle
            'azimuth': yaw,      # Horizontal angle
            'roll': roll,        # Tilt (should be ~0 for level mount)
            'wx': wx,
            'wy': wy,
            'wz': wz
        }
    
    def check_movement(self, data):
        """Check if dish is moving (wind, etc.)"""
        # Check if any rotation rate exceeds threshold
        max_rotation = max(abs(data['wx']), abs(data['wy']), abs(data['wz']))
        return max_rotation > MOVEMENT_THRESHOLD
    
    def notification_handler(self, sender, data):
        """Handle incoming data"""
        parsed = self.parse_data(data)
        if not parsed:
            return
        
        self.elevation = parsed['elevation']
        self.azimuth = parsed['azimuth']
        self.movement_detected = self.check_movement(parsed)
        
        # Display current position
        status = "⚠️  MOVING!" if self.movement_detected else "✓ Stable"
        
        print(f"{status} | Elevation: {self.elevation:6.2f}° | Azimuth: {self.azimuth:6.2f}° | "
              f"Roll: {parsed['roll']:5.2f}° | Rotation: {max(abs(parsed['wx']), abs(parsed['wy']), abs(parsed['wz'])):5.1f}°/s")
    
    async def connect(self):
        """Connect to the sensor"""
        print(f"Connecting to dish sensor at {self.address}...")
        self.client = BleakClient(self.address)
        await self.client.connect()
        print("Connected!\n")
        
        # Start receiving notifications
        await self.client.start_notify(self.notify_char, self.notification_handler)
        
        # Send initialization commands
        print("Initializing sensor...")
        
        # Unlock register
        unlock_cmd = bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5])
        await self.client.write_gatt_char(self.write_char, unlock_cmd, response=False)
        await asyncio.sleep(0.1)
        
        # Set return rate to 1Hz (once per second)
        rate_cmd = bytes([0xFF, 0xAA, 0x03, 0x03, 0x00])
        await self.client.write_gatt_char(self.write_char, rate_cmd, response=False)
        await asyncio.sleep(0.1)
        
        # Save settings
        save_cmd = bytes([0xFF, 0xAA, 0x00, 0x00, 0x00])
        await self.client.write_gatt_char(self.write_char, save_cmd, response=False)
        await asyncio.sleep(0.5)
        
        print("Monitoring dish position... (Press Ctrl+C to stop)\n")
        print("Legend:")
        print("  Elevation = Vertical pointing angle (pitch)")
        print("  Azimuth   = Horizontal pointing angle (yaw)")
        print("  Roll      = Tilt (should be ~0° for level mount)")
        print("  Rotation  = Movement speed (movement detected if > 5°/s)\n")
        
    async def disconnect(self):
        """Disconnect from the sensor"""
        if self.client and self.client.is_connected:
            try:
                await self.client.stop_notify(self.notify_char)
            except:
                pass
            await self.client.disconnect()
            print("\nDisconnected")
    
    def get_position(self):
        """Get current dish position"""
        return {
            'elevation': self.elevation,
            'azimuth': self.azimuth,
            'is_moving': self.movement_detected
        }

async def main():
    monitor = DishMonitor(SENSOR_MAC)
    
    try:
        await monitor.connect()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await monitor.disconnect()
        
        # Print final position
        pos = monitor.get_position()
        print(f"\nFinal Position:")
        print(f"  Elevation: {pos['elevation']:.2f}°")
        print(f"  Azimuth: {pos['azimuth']:.2f}°")

if __name__ == "__main__":
    asyncio.run(main())