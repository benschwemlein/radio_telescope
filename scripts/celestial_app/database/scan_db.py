"""
Database module for storing radio telescope scans.
Uses SQLite for simple, file-based storage.
"""
import sqlite3
import os
from datetime import datetime
from typing import List, Optional, Dict, Any

# Database file location
DB_FILE = "scans.db"


class ScanDatabase:
    """Manages radio telescope scan data in SQLite database."""
    
    def __init__(self, db_path: str = DB_FILE):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                altitude REAL NOT NULL,
                azimuth REAL NOT NULL,
                duration_seconds REAL NOT NULL,
                resolution REAL NOT NULL,
                start_time TEXT NOT NULL,
                created_at TEXT NOT NULL,
                notes TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_scan(self, name: str, altitude: float, azimuth: float,
                 duration_seconds: float, resolution: float,
                 start_time: datetime, notes: str = "") -> int:
        """
        Add a new scan to the database.
        
        Args:
            name: Scan identifier/name
            altitude: Telescope altitude angle in degrees (0-90)
            azimuth: Telescope azimuth angle in degrees (0-360)
            duration_seconds: Scan duration in seconds
            resolution: Telescope beam width in degrees
            start_time: Scan start time as datetime object
            notes: Optional notes about the scan
        
        Returns:
            ID of the newly created scan
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        created_at = datetime.now().isoformat()
        start_time_str = start_time.isoformat()
        
        cursor.execute("""
            INSERT INTO scans (name, altitude, azimuth, duration_seconds, 
                             resolution, start_time, created_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, altitude, azimuth, duration_seconds, resolution,
              start_time_str, created_at, notes))
        
        scan_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return scan_id
    
    def get_all_scans(self) -> List[Dict[str, Any]]:
        """
        Retrieve all scans from the database.
        
        Returns:
            List of dictionaries containing scan data
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, altitude, azimuth, duration_seconds,
                   resolution, start_time, created_at, notes
            FROM scans
            ORDER BY start_time DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to list of dicts
        scans = []
        for row in rows:
            scan = {
                'id': row['id'],
                'name': row['name'],
                'altitude': row['altitude'],
                'azimuth': row['azimuth'],
                'duration_seconds': row['duration_seconds'],
                'resolution': row['resolution'],
                'start_time': datetime.fromisoformat(row['start_time']),
                'created_at': datetime.fromisoformat(row['created_at']),
                'notes': row['notes'] or ""
            }
            scans.append(scan)
        
        return scans
    
    def get_scan_by_id(self, scan_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific scan by ID.
        
        Args:
            scan_id: Scan ID to retrieve
        
        Returns:
            Dictionary with scan data, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, altitude, azimuth, duration_seconds,
                   resolution, start_time, created_at, notes
            FROM scans
            WHERE id = ?
        """, (scan_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return {
            'id': row['id'],
            'name': row['name'],
            'altitude': row['altitude'],
            'azimuth': row['azimuth'],
            'duration_seconds': row['duration_seconds'],
            'resolution': row['resolution'],
            'start_time': datetime.fromisoformat(row['start_time']),
            'created_at': datetime.fromisoformat(row['created_at']),
            'notes': row['notes'] or ""
        }
    
    def delete_scan(self, scan_id: int) -> bool:
        """
        Delete a scan from the database.
        
        Args:
            scan_id: ID of scan to delete
        
        Returns:
            True if scan was deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted
    
    def update_scan(self, scan_id: int, **kwargs) -> bool:
        """
        Update scan fields.
        
        Args:
            scan_id: ID of scan to update
            **kwargs: Fields to update (name, altitude, azimuth, etc.)
        
        Returns:
            True if scan was updated, False if not found
        """
        allowed_fields = ['name', 'altitude', 'azimuth', 'duration_seconds',
                         'resolution', 'start_time', 'notes']
        
        updates = []
        values = []
        
        for field, value in kwargs.items():
            if field in allowed_fields:
                updates.append(f"{field} = ?")
                # Convert datetime to string if needed
                if field == 'start_time' and isinstance(value, datetime):
                    value = value.isoformat()
                values.append(value)
        
        if not updates:
            return False
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = f"UPDATE scans SET {', '.join(updates)} WHERE id = ?"
        values.append(scan_id)
        
        cursor.execute(query, values)
        updated = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return updated
