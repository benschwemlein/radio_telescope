"""
Centralized color theme management for UI components.
"""

class ColorTheme:
    """Color theme definitions for dark and light modes."""
    
    DARK = {
        'bg_color': '#0a0a0e',
        'fg_color': 'white',
        'grid_color': '#404050',
        'accent_color': '#ff6060',
        'secondary_color': '#8080a0'
    }
    
    LIGHT = {
        'bg_color': 'white',
        'fg_color': 'black',
        'grid_color': '#c0c0c0',
        'accent_color': '#ff0000',
        'secondary_color': '#606060'
    }
    
    @staticmethod
    def get_theme(dark_mode: bool = True) -> dict:
        """
        Get color theme based on mode.
        
        Args:
            dark_mode: If True, return dark theme; otherwise light theme
        
        Returns:
            Dictionary with color definitions
        """
        return ColorTheme.DARK.copy() if dark_mode else ColorTheme.LIGHT.copy()
    
    @staticmethod
    def get_color(dark_mode: bool, color_key: str) -> str:
        """
        Get a specific color from the theme.
        
        Args:
            dark_mode: Theme mode
            color_key: Key name (e.g., 'bg_color', 'fg_color')
        
        Returns:
            Color string
        """
        theme = ColorTheme.get_theme(dark_mode)
        return theme.get(color_key, theme['fg_color'])
