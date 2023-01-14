import pygame
from pygame.locals import *
import sys  # 외장 모듈
import math

WIDTH = 1500
HEIGHT = 1200
FPS = 120

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

PIXEL_PER_METER = (10.0 / 1.0)  # 10 pixel 1m
# RUN_SPEED_KMPH = 0.6  # Km / Hour
# RUN_SPEED_MPM = (RUN_SPEED_KMPH * 1000.0 / 60.0)
# RUN_SPEED_MPS = (RUN_SPEED_MPM / 60.0)
# RUN_SPEED_PPS = int(RUN_SPEED_MPS * PIXEL_PER_METER)
# FRICTION = int(RUN_SPEED_MPS / 3 * PIXEL_PER_METER * 30)


def key_event():
    for event in pygame.event.get():
        if event.type == QUIT:
            print("quit")
            pygame.quit()
            sys.exit()


def get_pixel(length):
    return int(length * PIXEL_PER_METER)


def get_display_pos(x_or_y, width_or_length):
    return x_or_y - width_or_length / 2


def draw_road():
    for road_name, road_info in road_dict.items():
        if road_info['direction'] == 'e' or road_info['direction'] == 'w':
            width = get_pixel(road_info['length'])
            height = get_pixel(road_info['width'])
        elif road_info['direction'] == 's' or road_info['direction'] == 'n':
            width = get_pixel(road_info['width'])
            height = get_pixel(road_info['length'])
        else:
            print("error by making roads: maybe wrong direction")
            return

        x = get_display_pos(road_info['x'], width)
        y = get_display_pos(road_info['y'], height)
        pygame.draw.rect(main_display, BLACK, (x, y, width, height))

        color = RED
        end_line_size = 3
        font = pygame.font.SysFont('malgungothic', 12)
        text = font.render(str(road_name), True, WHITE)
        if road_info['direction'] == 'e':
            pygame.draw.rect(main_display, color, (x + width - end_line_size, y, end_line_size, height))
            main_display.blit(text, (x + width - 20, y))
        elif road_info['direction'] == 'w':
            pygame.draw.rect(main_display, color, (x, y, end_line_size, height))
            main_display.blit(text, (x + 10, y))
        elif road_info['direction'] == 's':
            pygame.draw.rect(main_display, color, (x, y + height - end_line_size, width, end_line_size))
            main_display.blit(text, (x + 3, y + height - 20))
        elif road_info['direction'] == 'n':
            pygame.draw.rect(main_display, color, (x, y, width, end_line_size))
            main_display.blit(text, (x + 3, y))


def get_road_point(road_name, point):
    road_info = road_dict[road_name]
    if point == "start":
        if road_info['direction'] == 'e':
            return [road_info['x'] - get_pixel(road_info['length']) / 2, road_info['y']]
        elif road_info['direction'] == 'w':
            return [road_info['x'] + get_pixel(road_info['length']) / 2, road_info['y']]
        elif road_info['direction'] == 's':
            return [road_info['x'], road_info['y'] - get_pixel(road_info['length']) / 2]
        elif road_info['direction'] == 'n':
            return [road_info['x'], road_info['y'] + get_pixel(road_info['length']) / 2]
    elif point == "end":
        if road_info['direction'] == 'e':
            return [road_info['x'] + get_pixel(road_info['length']) / 2, road_info['y']]
        elif road_info['direction'] == 'w':
            return [road_info['x'] - get_pixel(road_info['length']) / 2, road_info['y']]
        elif road_info['direction'] == 's':
            return [road_info['x'], road_info['y'] + get_pixel(road_info['length']) / 2]
        elif road_info['direction'] == 'n':
            return [road_info['x'], road_info['y'] - get_pixel(road_info['length']) / 2]
    else:
        print("error by get road point: maybe wrong point")
        return


def set_car_start_point():
    for car_name, car_info in car_dict.items():
        car_info['x'], car_info['y'] = get_road_point(car_info['Route'][0], "start")
        car_info['current_road'] = car_info['Route'][0]


def collide_check(point1, point2, radius):
    if abs(point1[0] - point2[0]) < radius and abs(point1[1] - point2[1]) < radius:
        return True
    else:
        return False


def draw_car():
    for car_name, car_info in car_dict.items():
        x = car_info['x']
        y = car_info['y']
        # 길 끝나면 다음길로 이동하는 알고리즘 생각하기
        # print(f"car: {car_name}, curr_road: {car_info['current_road']}, {car_info['crossing']}")

        if car_info['crossing'] is True:    # 길이 교차로일 때
            try:
                if collide_check([x, y], get_road_point(car_info['Route'][car_info['Route'].index(car_info['current_road']) + 1], "start"), 5):
                    car_info['crossing'] = False
                    car_info['current_road'] = car_info['Route'][car_info['Route'].index(car_info['current_road']) + 1]
                    car_info['x'], car_info['y'] = get_road_point(car_info['current_road'], "start")
                    x = car_info['x']
                    y = car_info['y']
                else:
                    print(get_road_point(car_info['Route'][car_info['Route'].index(car_info['current_road'])], "end"), get_road_point(car_info['Route'][car_info['Route'].index(car_info['current_road']) + 1], "start"))
                    x += (get_road_point(car_info['Route'][car_info['Route'].index(car_info['current_road']) + 1], "start")[0] - get_road_point(car_info['current_road'], "end")[0]) / 100
                    y += (get_road_point(car_info['Route'][car_info['Route'].index(car_info['current_road']) + 1], "start")[1] - get_road_point(car_info['current_road'], "end")[1]) / 100
            except Exception:
                x, y = get_road_point(car_info['Route'][0], "start")
                car_info['current_road'] = car_info['Route'][0]
                car_info['crossing'] = False

        else:   # crossing이 False일 때
            if collide_check([x, y], get_road_point(car_info['current_road'], "end"), 5):
                x, y = get_road_point(car_info['current_road'], "end")
                print("end")
                car_info['crossing'] = True

            if road_dict[car_info['current_road']]['direction'] == 'e':
                x += get_pixel(car_info['speed'])
            if road_dict[car_info['current_road']]['direction'] == 'w':
                x -= get_pixel(car_info['speed'])
            if road_dict[car_info['current_road']]['direction'] == 's':
                y += get_pixel(car_info['speed'])
            if road_dict[car_info['current_road']]['direction'] == 'n':
                y -= get_pixel(car_info['speed'])

        car_info['x'] = x
        car_info['y'] = y

        color = 0
        if car_name == 1:
            color = RED
        else:
            color = BLUE
        pygame.draw.circle(main_display, color, (x, y), 10)


road_width = 2
road_length = 50

road_dict = {
    1: {"x": WIDTH/2 - get_pixel(25+2), "y": HEIGHT/2 - 11, "width": road_width, "length": road_length, "direction": "w"},
    1.1: {"x": WIDTH/2 - get_pixel(25+2), "y": HEIGHT/2 + 11, "width": road_width, "length": road_length, "direction": "e"},
    2: {"x": WIDTH/2 + get_pixel(25+2), "y": HEIGHT/2 - 11, "width": road_width, "length": road_length, "direction": "w"},
    2.1: {"x": WIDTH/2 + get_pixel(25+2), "y": HEIGHT/2 + 11, "width": road_width, "length": road_length, "direction": "e"},

    3: {"x": WIDTH/2 - 11, "y": HEIGHT/2 - get_pixel(25+2), "width": road_width, "length": road_length, "direction": "s"},
    3.1: {"x": WIDTH/2 + 11, "y": HEIGHT/2 - get_pixel(25+2), "width": road_width, "length": road_length, "direction": "n"},
    4: {"x": WIDTH/2 - 11, "y": HEIGHT/2 + get_pixel(25+2), "width": road_width, "length": road_length, "direction": "s"},
    4.1: {"x": WIDTH/2 + 11, "y": HEIGHT/2 + get_pixel(25+2), "width": road_width, "length": road_length, "direction": "n"},
}

car_dict = {
    1: {"x": 0.0, "y": 0.0, "crossing": False, "current_road": 0.0, "speed": 0.1, "Route": [4.1, 2.1, 2, 3.1, 3, 4]},
    # 2: {"x": 0.0, "y": 0.0, "crossing": False, "current_road": 0.0, "speed": 0.1, "Route": [1.1, 2.1]},
    # 3: {"x": 0.0, "y": 0.0, "crossing": False, "current_road": 0.0, "speed": 0.1, "Route": [2, 1]},

}


pygame.init()  # 초기화
pygame.display.set_caption('Traffic Simulator')
main_display = pygame.display.set_mode((WIDTH, HEIGHT), 0)
clock = pygame.time.Clock()  # 시간 설정
set_car_start_point()
while True:
    key_event()
    main_display.fill(WHITE)

    draw_road()
    draw_car()

    pygame.display.update()  # 화면을 업데이트한다
    clock.tick(FPS)  # 화면 표시 회수 설정만큼 루프의 간격을 둔다
