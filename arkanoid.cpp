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
    void runMainLoop();

    static void mainloop(void* arg);

private:
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    GameObject paddle;
    GameObject ball;
    std::vector<GameObject> bricks;
    int ballSpeedX = 5;
    int ballSpeedY = -5;
    int windowWidth = SCREEN_WIDTH;
    int windowHeight = SCREEN_HEIGHT;
    bool quit = false;

    void createBricks();
    void scaleBricks();
};

Game::Game() {
    srand(static_cast<unsigned>(time(nullptr)));
}

Game::~Game() {
    if (renderer) {
        SDL_DestroyRenderer(renderer);
    }
    if (window) {
        SDL_DestroyWindow(window);
    }
    SDL_Quit();
}

bool Game::init() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        return false;
    }

    SDL_DisplayMode current;
    SDL_GetCurrentDisplayMode(0, &current);

    window = SDL_CreateWindow("Arkanoid WASM", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 
                              current.w, current.h, SDL_WINDOW_SHOWN);
    if (!window) {
        return false;
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        return false;
    }

    SDL_GetWindowSize(window, &windowWidth, &windowHeight);

    float scaleX = static_cast<float>(windowWidth) / SCREEN_WIDTH;
    float scaleY = static_cast<float>(windowHeight) / SCREEN_HEIGHT;

    paddle.rect = {windowWidth / 2 - static_cast<int>(PADDLE_WIDTH * scaleX) / 2, 
                   windowHeight - static_cast<int>(PADDLE_HEIGHT * scaleY) - 10, 
                   static_cast<int>(PADDLE_WIDTH * scaleX), 
                   static_cast<int>(PADDLE_HEIGHT * scaleY)};
    paddle.color = {255, 255, 255, 255};

    ball.rect = {windowWidth / 2 - static_cast<int>(BALL_SIZE * scaleX) / 2, 
                 windowHeight / 2 - static_cast<int>(BALL_SIZE * scaleY) / 2, 
                 static_cast<int>(BALL_SIZE * scaleX), 
                 static_cast<int>(BALL_SIZE * scaleY)};
    ball.color = {255, 255, 255, 255};

    createBricks();
    scaleBricks();

    return true;
}

void Game::createBricks() {
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 10; ++x) {
            GameObject brick;
            brick.rect = {x * (BRICK_WIDTH + 1), y * (BRICK_HEIGHT + 1) + 50, BRICK_WIDTH, BRICK_HEIGHT};
            brick.color = {static_cast<Uint8>(rand() % 256), 
                           static_cast<Uint8>(rand() % 256), 
                           static_cast<Uint8>(rand() % 256), 255};
            bricks.push_back(brick);
        }
    }
}

void Game::scaleBricks() {
    float scaleX = static_cast<float>(windowWidth) / SCREEN_WIDTH;
    float scaleY = static_cast<float>(windowHeight) / SCREEN_HEIGHT;

    for (auto& brick : bricks) {
        brick.rect.x = static_cast<int>(brick.rect.x * scaleX);
        brick.rect.y = static_cast<int>(brick.rect.y * scaleY);
        brick.rect.w = static_cast<int>(brick.rect.w * scaleX);
        brick.rect.h = static_cast<int>(brick.rect.h * scaleY);
    }
}

void Game::handleEvents() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) {
            quit = true;
        }
    }

    int mouseX, mouseY;
    SDL_GetMouseState(&mouseX, &mouseY);

    // Use the actual window width for paddle movement
    paddle.rect.x = mouseX - paddle.rect.w / 2;
    if (paddle.rect.x < 0) paddle.rect.x = 0;
    if (paddle.rect.x > windowWidth - paddle.rect.w) paddle.rect.x = windowWidth - paddle.rect.w;
}

void Game::update() {
    ball.rect.x += ballSpeedX;
    ball.rect.y += ballSpeedY;

    if (ball.rect.x <= 0 || ball.rect.x + ball.rect.w >= windowWidth) ballSpeedX = -ballSpeedX;
    if (ball.rect.y <= 0) ballSpeedY = -ballSpeedY;
    if (ball.rect.y + ball.rect.h >= windowHeight) {
        ball.rect.x = windowWidth / 2 - ball.rect.w / 2;
        ball.rect.y = windowHeight / 2 - ball.rect.h / 2;
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

void Game::mainloop(void* arg) {
    Game* game = static_cast<Game*>(arg);
    game->handleEvents();
    game->update();
    game->render();
    
    if (game->quit) {
        emscripten_cancel_main_loop();
    }
}

void Game::runMainLoop() {
    emscripten_set_main_loop_arg(Game::mainloop, this, 0, 1);
}

int main() {
    Game game;
    if (!game.init()) {
        return 1;
    }
    game.runMainLoop();
    return 0;
}
