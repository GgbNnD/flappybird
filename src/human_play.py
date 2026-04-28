import gymnasium
import numpy as np
import pygame
import flappy_bird_gymnasium
from pygame.locals import *

"""
运行python human_play.py可以试玩一下Flappy Bird
可以通过Obs的输出验证一下各个观测值在画面中的位置
"""

# 全局屏幕设置
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512

def draw_text(surface, text, size, x, y, color=(255, 255, 255)):
    font = pygame.font.SysFont("Arial", size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surface.blit(text_surface, text_rect)

def show_start_screen(screen):
    screen.fill((0, 0, 0))
    draw_text(screen, "Flappy Bird", 50, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4)
    draw_text(screen, "Press SPACE to Start", 22, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    draw_text(screen, "or UP Arrow to Jump", 22, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40)
    pygame.display.flip()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_SPACE or event.key == pygame.K_UP):
                waiting = False
    return True

def show_game_over_screen(screen, score):
    screen.fill((0, 0, 0))  # 清屏为黑色背景
    
    # 计算各元素位置
    title_y = SCREEN_HEIGHT // 6  # 标题位置上移
    score_y = SCREEN_HEIGHT // 3   # 分数位置上移
    options_y = SCREEN_HEIGHT * 2 // 3  # 选项位置下移
    
    # 绘制游戏结束标题
    draw_text(screen, "GAME OVER", 30, SCREEN_WIDTH // 2, title_y, (255, 50, 50))
    
    # 绘制分数
    draw_text(screen, f"YOUR SCORE: {score}", 24, SCREEN_WIDTH // 2, score_y, (255, 255, 255))
    
    # 绘制选项
    draw_text(screen, "Press [R] to Restart", 18, SCREEN_WIDTH // 2, options_y, (200, 200, 200))
    draw_text(screen, "Press [Q] to Quit", 18, SCREEN_WIDTH // 2, options_y + 40, (200, 200, 200))
    
    pygame.display.flip()  # 更新显示
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return True  # 重新开始
                elif event.key == pygame.K_q:
                    return False  # 退出游戏
    return False

def play(use_lidar=True):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird")
    
    while True:
        # 显示开始界面
        if not show_start_screen(screen):
            break
            
        # 初始化游戏环境
        env = gymnasium.make(
            "FlappyBird-v0", audio_on=False, render_mode="human", use_lidar=use_lidar
        )
        
        steps = 0
        video_buffer = []
        obs = env.reset()
        score = 0
        
        # 游戏主循环
        running = True
        while running:
            # 获取动作输入
            action = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and (
                    event.key == pygame.K_SPACE or event.key == pygame.K_UP
                ):
                    action = 1

            # 游戏步进
            obs, reward, done, _, info = env.step(action)
            # print(f"done: {done}")
            video_buffer.append(obs)
            score = info['score']
            
            steps += 1
            print(f"reward: {reward}")
            print(
                f"Obs: {obs}\n"
                f"Action: {action}\n"
                f"Score: {score}\n Steps: {steps}\n"
            )

            if done:
                running = False
        
        
        # 显示结束界面
        if not show_game_over_screen(screen, score):
            break

    pygame.quit()

if __name__ == "__main__":
    play(use_lidar=False)