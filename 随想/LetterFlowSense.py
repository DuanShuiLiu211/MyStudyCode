import pygame
import random


# 创建屏幕窗口的黑色画布
pygame.init()
pygame.display.set_caption('LetterFlowSense')
PANEL_width = 1080
PANEL_height = 720
winSur = pygame.display.set_mode((PANEL_width, PANEL_height), flags=pygame.RESIZABLE)
winSur.fill((0, 0, 0))
# 创建黑色半透明背景图布
bg_suface = pygame.Surface((PANEL_width, PANEL_height), flags=pygame.SRCALPHA)
bg_suface.fill(pygame.Color(0, 0, 0, 25))

# 设置字母集与字体以及字号
FONT_px = 20
letter = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c',
          'v', 'b', 'n', 'm']
font = pygame.font.SysFont('/Users/WangHao/Library/Fonts/timr45w.ttf', FONT_px)
# 根据字号与屏幕窗口画布宽度计算总列数并设置行起始值
drops = [0 for i in range(int(PANEL_width / FONT_px))]

# 程序主循环
while True:
    # 从队列中获取事件
    for event in pygame.event.get():
        # 按下关闭按钮中止
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.KEYDOWN:
            keybool = list(pygame.key.get_pressed())
            # 按下esc键中止
            if keybool[41]:
                exit()

    # 设置时间延迟
    pygame.time.delay(30)

    # 应用黑色半透明背景图布于屏幕窗口画布
    winSur.blit(bg_suface, (0, 0))
    # 获取当前屏幕窗口画布尺寸并相应调整
    current_width, current_height = pygame.display.Info().current_w, pygame.display.Info().current_h
    if PANEL_width != current_width:
        drops = [0 for i in range(int(current_width / FONT_px))]
        bg_suface = pygame.Surface((current_width, current_height), flags=pygame.SRCALPHA)
        bg_suface.fill(pygame.Color(0, 0, 0, 25))
        PANEL_width = current_width

    # 绘制字母流图像
    for i in range(len(drops)):
        text = random.choice([font.render(str(letter[i]), True, (0, 255, 0)) for i in range(26)])
        winSur.blit(text, (i * FONT_px, drops[i] * FONT_px))
        drops[i] += 1
        if drops[i] * 10 > current_height or random.random() > 0.95:
            drops[i] = 0
    # 更新待显示的画布到屏幕
    pygame.display.flip()
