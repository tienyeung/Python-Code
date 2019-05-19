class Settings():
    def __init__(self):
        """初始化游戏静态设置"""
        self.screen_width=1200
        self.screen_height=700
        self.bg_color=(230,230,230)

        #bullet
        self.bullet_width=300
        self.bullet_height=15
        self.bullet_color=60,60,60
        self.bullets_allowed=3

        #外星人
        self.alien_speed_factor=1#水平速度
        self.fleet_drop_speed=1#垂直速度
        self.fleet_direction=1#1为右移

        #飞船
        self.ship_limit=3

        #加速游戏节奏
        self.speedup_scale=1.1
        #外星人点数的提高速度
        self.score_scale=1.5

        self.initialize_dynamic_settings()

    def initialize_dynamic_settings(self):
        self.ship_speed_factor=1.5
        self.bullet_speed_factor=3
        self.alien_speed_factor=1
        self.fleet_direction=1
        #计分
        self.alien_point=50

    def increase_speed(self):
        self.ship_speed_factor *= self.speedup_scale
        self.bullet_speed_factor *= self.speedup_scale
        self.alien_speed_factor *= self.speedup_scale

        self.alien_points=int(self.alien_points*self.score_scale)
