// arkanoid.cpp
#include <emscripten.h>
#include <SDL2/SDL.h>
#include <vector>
#include <cstdlib>
#include <ctime>

const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;
const int PADDLE_WIDTH = 100;
const int PADDLE_HEIGHT = 20;
const int BALL_SIZE = 15;
const int BRICK_WIDTH = 80;
const int BRICK_HEIGHT = 30;

struct GameObject {
    SDL_Rect rect;
    SDL_Color color;
    bool active = true;
};

class Game {
public:
    Game();
    ~Game();
    bool init();
    void handleEvents();
    void update();
    void render();

private:
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    GameObject paddle;
    GameObject ball;
    std::vector<GameObject> bricks;
    int ballSpeedX = 5;
    int ballSpeedY = -5;

    void createBricks();
};

Game* game = nullptr;

Game::Game() {
    srand(static_cast<unsigned>(time(nullptr)));
}

Game::~Game() {
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

bool Game::init() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        return false;
    }

    window = SDL_CreateWindow("Arkanoid WASM", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) {
        return false;
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        return false;
    }

    paddle.rect = {SCREEN_WIDTH / 2 - PADDLE_WIDTH / 2, SCREEN_HEIGHT - PADDLE_HEIGHT - 10, PADDLE_WIDTH, PADDLE_HEIGHT};
    paddle.color = {255, 255, 255, 255};

    ball.rect = {SCREEN_WIDTH / 2 - BALL_SIZE / 2, SCREEN_HEIGHT / 2 - BALL_SIZE / 2, BALL_SIZE, BALL_SIZE};
    ball.color = {255, 255, 255, 255};

    createBricks();

    return true;
}

void Game::createBricks() {
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 10; ++x) {
            GameObject brick;
            brick.rect = {x * (BRICK_WIDTH + 1), y * (BRICK_HEIGHT + 1) + 50, BRICK_WIDTH, BRICK_HEIGHT};
            brick.color = {static_cast<Uint8>(rand() % 256), static_cast<Uint8>(rand() % 256), static_cast<Uint8>(rand() % 256), 255};
            bricks.push_back(brick);
        }
    }
}

void Game::handleEvents() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) {
            emscripten_cancel_main_loop();
        }
    }

    int mouseX, mouseY;
    SDL_GetMouseState(&mouseX, &mouseY);
    paddle.rect.x = mouseX - PADDLE_WIDTH / 2;
    if (paddle.rect.x < 0) paddle.rect.x = 0;
    if (paddle.rect.x > SCREEN_WIDTH - PADDLE_WIDTH) paddle.rect.x = SCREEN_WIDTH - PADDLE_WIDTH;
}

void Game::update() {
    ball.rect.x += ballSpeedX;
    ball.rect.y += ballSpeedY;

    if (ball.rect.x <= 0 || ball.rect.x + BALL_SIZE >= SCREEN_WIDTH) ballSpeedX = -ballSpeedX;
    if (ball.rect.y <= 0) ballSpeedY = -ballSpeedY;
    if (ball.rect.y + BALL_SIZE >= SCREEN_HEIGHT) {
        // Reset ball position instead of quitting
        ball.rect.x = SCREEN_WIDTH / 2 - BALL_SIZE / 2;
        ball.rect.y = SCREEN_HEIGHT / 2 - BALL_SIZE / 2;
        ballSpeedY = -ballSpeedY;
    }

    if (SDL_HasIntersection(&ball.rect, &paddle.rect)) {
        ballSpeedY = -ballSpeedY;
    }

    for (auto& brick : bricks) {
        if (brick.active && SDL_HasIntersection(&ball.rect, &brick.rect)) {
            brick.active = false;
            ballSpeedY = -ballSpeedY;
            break;
        }
    }
}

void Game::render() {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    SDL_SetRenderDrawColor(renderer, paddle.color.r, paddle.color.g, paddle.color.b, paddle.color.a);
    SDL_RenderFillRect(renderer, &paddle.rect);

    SDL_SetRenderDrawColor(renderer, ball.color.r, ball.color.g, ball.color.b, ball.color.a);
    SDL_RenderFillRect(renderer, &ball.rect);

    for (const auto& brick : bricks) {
        if (brick.active) {
            SDL_SetRenderDrawColor(renderer, brick.color.r, brick.color.g, brick.color.b, brick.color.a);
            SDL_RenderFillRect(renderer, &brick.rect);
        }
    }

    SDL_RenderPresent(renderer);
}

void mainloop() {
    game->handleEvents();
    game->update();
    game->render();
}

int main() {
    game = new Game();
    if (!game->init()) {
        delete game;
        return 1;
    }

    emscripten_set_main_loop(mainloop, 0, 1);

    return 0;
}
