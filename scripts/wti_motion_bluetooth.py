#!/usr/bin/env python3
"""
WT901BLECL BLE Sensor Reader for Raspberry Pi
Automatically discovers characteristics and reads sensor data
"""

import asyncio
import struct
from bleak import BleakClient, BleakScanner

# Your sensor's MAC address
SENSOR_MAC = "E0:D6:FC:57:08:EF"

class WT901BLE:
    def __init__(self, address):
        self.address = address
        self.client = None
        self.notify_char = None
        self.write_char = None
        
    def parse_data(self, data):
        """Parse the sensor data packet (20 bytes combined format)"""
        if len(data) != 20:
            return None
            
        # Check header
        if data[0] != 0x55 or data[1] != 0x61:
            return None
        
        # Parse acceleration (bytes 2-7)
        ax = struct.unpack('<h', data[2:4])[0] / 32768.0 * 16  # g
        ay = struct.unpack('<h', data[4:6])[0] / 32768.0 * 16
        az = struct.unpack('<h', data[6:8])[0] / 32768.0 * 16
        
        # Parse angular velocity (bytes 8-13)
        wx = struct.unpack('<h', data[8:10])[0] / 32768.0 * 2000  # °/s
        wy = struct.unpack('<h', data[10:12])[0] / 32768.0 * 2000
        wz = struct.unpack('<h', data[12:14])[0] / 32768.0 * 2000
        
        # Parse angles (bytes 14-19)
        roll = struct.unpack('<h', data[14:16])[0] / 32768.0 * 180  # degrees
        pitch = struct.unpack('<h', data[16:18])[0] / 32768.0 * 180
        yaw = struct.unpack('<h', data[18:20])[0] / 32768.0 * 180
        
        result = {
            'type': 'combined',
            'ax': ax,
            'ay': ay,
            'az': az,
            'wx': wx,
            'wy': wy,
            'wz': wz,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw
        }
            
        return result
    
    def notification_handler(self, sender, data):
        """Handle incoming BLE notifications"""
        parsed = self.parse_data(data)
        if parsed:
            # Print all data on one line for compact display
            print(f"Accel: X={parsed['ax']:6.3f}g Y={parsed['ay']:6.3f}g Z={parsed['az']:6.3f}g | "
                  f"Gyro: X={parsed['wx']:7.1f}°/s Y={parsed['wy']:7.1f}°/s Z={parsed['wz']:7.1f}°/s | "
                  f"Angle: R={parsed['roll']:6.2f}° P={parsed['pitch']:6.2f}° Y={parsed['yaw']:6.2f}°")
        else:
            # Print raw data if parsing fails
            print(f"Raw data ({len(data)} bytes): {data.hex()}")
    
    async def discover_and_connect(self):
        """Discover characteristics and connect to the sensor"""
        print(f"Connecting to {self.address}...")
        self.client = BleakClient(self.address)
        await self.client.connect()
        print("Connected!\n")
        
        # Discover services and characteristics
        print("Discovering services and characteristics...")
        notify_candidates = []
        write_candidates = []
        
        for service in self.client.services:
            print(f"\nService: {service.uuid}")
            if service.description:
                print(f"  Description: {service.description}")
            
            for char in service.characteristics:
                print(f"  Characteristic: {char.uuid}")
                if char.description:
                    print(f"    Description: {char.description}")
                print(f"    Properties: {char.properties}")
                
                # Look for notify characteristic
                if "notify" in char.properties:
                    notify_candidates.append(char)
                    print(f"    *** This characteristic supports NOTIFY ***")
                
                # Look for write characteristic
                if "write" in char.properties or "write-without-response" in char.properties:
                    write_candidates.append(char)
                    print(f"    *** This characteristic supports WRITE ***")
        
        # Select the notify characteristic
        if not notify_candidates:
            print("\nERROR: No notify characteristics found!")
            return False
        
        # Try to find the most likely candidate
        for char in notify_candidates:
            if "ffe4" in char.uuid.lower():
                self.notify_char = char
                break
        
        if not self.notify_char:
            self.notify_char = notify_candidates[0]
        
        # Select write characteristic
        for char in write_candidates:
            if "ffe9" in char.uuid.lower():
                self.write_char = char
                break
        
        if not self.write_char and write_candidates:
            self.write_char = write_candidates[0]
        
        print(f"\n✓ Using notify characteristic: {self.notify_char.uuid}")
        if self.write_char:
            print(f"✓ Using write characteristic: {self.write_char.uuid}")
        
        # Start receiving notifications
        print("\nStarting notifications...")
        await self.client.start_notify(self.notify_char.uuid, self.notification_handler)
        
        # Send unlock command and request data
        if self.write_char:
            print("Sending unlock command...")
            # Unlock register: FF AA 69 88 B5
            unlock_cmd = bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5])
            await self.client.write_gatt_char(self.write_char.uuid, unlock_cmd, response=False)
            await asyncio.sleep(0.1)
            
            print("Requesting continuous data output...")
            # Set return rate to 10Hz: FF AA 03 06 00
            rate_cmd = bytes([0xFF, 0xAA, 0x03, 0x06, 0x00])
            await self.client.write_gatt_char(self.write_char.uuid, rate_cmd, response=False)
            await asyncio.sleep(0.1)
            
            # Save settings: FF AA 00 00 00
            save_cmd = bytes([0xFF, 0xAA, 0x00, 0x00, 0x00])
            await self.client.write_gatt_char(self.write_char.uuid, save_cmd, response=False)
            await asyncio.sleep(0.5)
        
        print("Reading data... (Press Ctrl+C to stop)\n")
        
        return True
        
    async def disconnect(self):
        """Disconnect from the sensor"""
        if self.client and self.client.is_connected:
            if self.notify_char:
                try:
                    await self.client.stop_notify(self.notify_char.uuid)
                except:
                    pass
            await self.client.disconnect()
            print("\nDisconnected")

async def main():
    sensor = WT901BLE(SENSOR_MAC)
    
    try:
        success = await sensor.discover_and_connect()
        if success:
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await sensor.disconnect()

if __name__ == "__main__":
    asyncio.run(main())