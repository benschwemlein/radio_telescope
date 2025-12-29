import pyqtgraph.opengl as gl

class FixedGLViewWidget(gl.GLViewWidget):
    """Custom GL view widget with fixed camera distance"""
    
    def wheelEvent(self, ev):
        """Disable zoom via wheel or trackpad scroll"""
        ev.ignore()
    
    def set_fixed_distance(self, d: float):
        """Set fixed camera distance"""
        self.opts["distance"] = float(d)
        self.update()