from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SoccerPitchConfiguration:
    """
    Configuration class for soccer pitch dimensions and layout.
    
    All measurements are in centimeters, following FIFA standards.
    The coordinate system has origin at bottom-left corner.
    """
    width: int = 7000  # [cm] - field width
    length: int = 12000  # [cm] - field length
    penalty_box_width: int = 4100  # [cm] - penalty area width
    penalty_box_length: int = 2015  # [cm] - penalty area length
    goal_box_width: int = 1832  # [cm] - goal area width
    goal_box_length: int = 550  # [cm] - goal area length
    centre_circle_radius: int = 915  # [cm] - center circle radius
    penalty_spot_distance: int = 1100  # [cm] - penalty spot from goal line

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        """
        Returns all key vertices of the soccer pitch for drawing and analysis.
        
        Returns:
            List[Tuple[int, int]]: List of (x, y) coordinates for all vertices
        """
        return [
            (0, 0),  # 1 - Bottom left corner
            (0, (self.width - self.penalty_box_width) / 2),  # 2 - Left penalty box bottom
            (0, (self.width - self.goal_box_width) / 2),  # 3 - Left goal box bottom
            (0, (self.width + self.goal_box_width) / 2),  # 4 - Left goal box top
            (0, (self.width + self.penalty_box_width) / 2),  # 5 - Left penalty box top
            (0, self.width),  # 6 - Top left corner
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7 - Left goal box front bottom
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8 - Left goal box front top
            (self.penalty_spot_distance, self.width / 2),  # 9 - Left penalty spot
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10 - Left penalty box front bottom
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11 - Left penalty box inner bottom
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12 - Left penalty box inner top
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13 - Left penalty box front top
            (self.length / 2, 0),  # 14 - Center bottom
            (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15 - Center circle bottom
            (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16 - Center circle top
            (self.length / 2, self.width),  # 17 - Center top
            (
                self.length - self.penalty_box_length,
                (self.width - self.penalty_box_width) / 2
            ),  # 18 - Right penalty box front bottom
            (
                self.length - self.penalty_box_length,
                (self.width - self.goal_box_width) / 2
            ),  # 19 - Right penalty box inner bottom
            (
                self.length - self.penalty_box_length,
                (self.width + self.goal_box_width) / 2
            ),  # 20 - Right penalty box inner top
            (
                self.length - self.penalty_box_length,
                (self.width + self.penalty_box_width) / 2
            ),  # 21 - Right penalty box front top
            (self.length - self.penalty_spot_distance, self.width / 2),  # 22 - Right penalty spot
            (
                self.length - self.goal_box_length,
                (self.width - self.goal_box_width) / 2
            ),  # 23 - Right goal box front bottom
            (
                self.length - self.goal_box_length,
                (self.width + self.goal_box_width) / 2
            ),  # 24 - Right goal box front top
            (self.length, 0),  # 25 - Bottom right corner
            (self.length, (self.width - self.penalty_box_width) / 2),  # 26 - Right penalty box bottom
            (self.length, (self.width - self.goal_box_width) / 2),  # 27 - Right goal box bottom
            (self.length, (self.width + self.goal_box_width) / 2),  # 28 - Right goal box top
            (self.length, (self.width + self.penalty_box_width) / 2),  # 29 - Right penalty box top
            (self.length, self.width),  # 30 - Top right corner
            (self.length / 2 - self.centre_circle_radius, self.width / 2),  # 31 - Center circle left
            (self.length / 2 + self.centre_circle_radius, self.width / 2),  # 32 - Center circle right
        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        # Outer field boundaries
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),  # Left side
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),  # Right side
        (1, 14), (14, 25),  # Bottom line
        (6, 17), (17, 30),  # Top line
        
        # Goal boxes
        (7, 8),  # Left goal box front
        (23, 24),  # Right goal box front
        (3, 7), (4, 8),  # Left goal box sides
        (27, 23), (28, 24),  # Right goal box sides
        
        # Penalty boxes
        (10, 11), (11, 12), (12, 13),  # Left penalty box front
        (18, 19), (19, 20), (20, 21),  # Right penalty box front
        (2, 10), (5, 13),  # Left penalty box sides
        (26, 18), (29, 21),  # Right penalty box sides
        
        # Center line and circle markers
        (14, 15), (15, 16), (16, 17),  # Center line segments
    ])

    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
        "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
        "14", "19"
    ])

    colors: List[str] = field(default_factory=lambda: [
        # Left side elements (pink/magenta)
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", 
        # Center elements (blue)
        "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", 
        # Right side elements (tomato/red)
        "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", 
        "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
        # Additional center elements
        "#00BFFF", "#00BFFF"
    ])