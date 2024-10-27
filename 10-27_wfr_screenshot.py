from __future__ import annotations
import copy
# Standard library imports
import os
import sys
import hashlib
import glob
from dataclasses import dataclass, field
from PIL import Image, ImageTk, ImageGrab
import base64
import io
import uuid
import json
import logging
import logging.handlers
import threading
import asyncio
from functools import partial
import platform
from datetime import datetime, timedelta
from collections import deque
from queue import Queue, Empty
from typing import Optional, Dict, List, TYPE_CHECKING

# GUI imports
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# Third-party imports
import win32gui
import win32con
import win32process
import psutil
import pyautogui
import anthropic
from pynput import mouse, keyboard
import pyperclip

if TYPE_CHECKING:
    from typing import Type
    from asyncio import AbstractEventLoop

# Optional: Configure logging at the start
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('workflow_analyzer.log', encoding='utf-8')
    ]
)
from enum import Enum, auto

class TaskCategory(Enum):
    AI_INTERACTION = auto()
    DEVELOPMENT = auto()
    ANALYSIS = auto()
    DOCUMENTATION = auto()
    PLANNING = auto()

class InteractionType(Enum):
    QUERY = auto()
    CODE_REVIEW = auto()
    BRAINSTORMING = auto()
    DEBUGGING = auto()
    CONVERSATION = auto()

class TaskContext:
    def __init__(self, category: TaskCategory, details: Optional[Dict] = None):
        self.category = category
        self.start_time = datetime.now()
        self.details = details or {}
        
    def to_dict(self) -> Dict:
        return {
            'category': self.category.name,
            'start_time': self.start_time.isoformat(),
            'details': self.details
        }

class ScreenshotSelector:
    def __init__(self, root):
        self.root = root
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.selection_window = None
        self.canvas = None
        self.screenshot = None
        self.selected_image = None
        
    def start_selection(self):
        """Create transparent fullscreen window for selection"""
        # Take full screenshot first
        self.screenshot = ImageGrab.grab()
        
        # Hide main window during selection
        self.root.iconify()
        
        # Create selection window
        self.selection_window = tk.Toplevel()
        self.selection_window.attributes('-alpha', 0.3, '-fullscreen', True)
        self.selection_window.attributes('-topmost', True)
        
        # Configure canvas for drawing selection rectangle
        self.canvas = tk.Canvas(self.selection_window, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Draw the screenshot as background
        self.tk_image = ImageTk.PhotoImage(self.screenshot)
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_move)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        
        # Bind escape to cancel
        self.selection_window.bind('<Escape>', lambda e: self._cancel_selection())
        
        # Wait for selection to complete
        self.root.wait_window(self.selection_window)
        
        # Return the selected image
        return self.selected_image
        
    def _on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        
    def _on_mouse_move(self, event):
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='red', width=2
        )
        
    def _on_mouse_up(self, event):
        if not self.start_x or not self.start_y:
            self._cancel_selection()
            return
            
        # Get coordinates
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        
        # Ensure minimum selection size
        if x2 - x1 < 10 or y2 - y1 < 10:
            self._cancel_selection()
            return
            
        # Crop the full screenshot
        self.selected_image = self.screenshot.crop((x1, y1, x2, y2))
        
        # Clean up and close selection window
        self.selection_window.destroy()
        self.root.deiconify()
        
    def _cancel_selection(self):
        """Cancel the selection process"""
        self.selected_image = None
        if self.selection_window:
            self.selection_window.destroy()
        self.root.deiconify()

def setup_logging():
    """Configure logging with proper encoding handling"""
    logger = logging.getLogger('WorkflowAnalyzer')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(console_handler)
    
    # File handler with UTF-8 encoding
    try:
        file_handler = logging.FileHandler(
            filename='workflow_analyzer.log',
            encoding='utf-8',
            mode='a'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to create file handler: {e}")
    
    return logger
@dataclass
class ContentContext:
    text_content: Optional[str] = None
    clipboard_content: Optional[str] = None
    document_name: Optional[str] = None
    document_type: Optional[str] = None
    
    def to_dict(self) -> Dict:
        # Hash sensitive content for privacy
        return {
            'text_content_hash': hashlib.sha256(self.text_content.encode()).hexdigest() if self.text_content else None,
            'clipboard_hash': hashlib.sha256(self.clipboard_content.encode()).hexdigest() if self.clipboard_content else None,
            'document_name': self.document_name,
            'document_type': self.document_type
        }
@dataclass
class WindowContext:
    app_name: str
    window_title: str
    process_id: int
    process_name: str
    focus_start: datetime
    focus_duration: timedelta = field(default_factory=lambda: timedelta(0))
    
    def update_duration(self, current_time: datetime) -> None:
        self.focus_duration = current_time - self.focus_start

@dataclass
class AIInteraction:
    platform: str
    interaction_type: str  # 'prompt', 'response', 'error'
    content_hash: str
    timestamp: datetime
    duration: timedelta
    context: Dict
    
@dataclass
class WorkflowEvent:
    def __init__(self, timestamp: datetime, event_type: str, window_title: str, details: Dict, task_context: Optional[TaskContext] = None):
        self.timestamp = timestamp
        self.event_type = event_type
        self.window_title = window_title
        self.details = details
        self.task_context = task_context

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'window_title': self.window_title,
            'details': self.details,
            'task_context': self.task_context.to_dict() if self.task_context else None
        }
# Add this new class for session management
class SessionManager:
    def __init__(self, base_dir="sessions"):
        self.base_dir = base_dir
        self.current_session = None
        
        # Ensure base directory exists
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
    def create_session(self, name):
        """Create a new session directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{name}_{timestamp}"
        session_dir = os.path.join(self.base_dir, session_id)
        
        # Create session directory structure
        os.makedirs(session_dir)
        os.makedirs(os.path.join(session_dir, "screenshots"))
        
        self.current_session = {
            'id': session_id,
            'name': name,
            'dir': session_dir,
            'start_time': datetime.now(),
            'screenshots': []  # Will only contain user-initiated screenshots
        }
        return self.current_session
        
    def save_session(self, events, conversation_history):
        """Save session data and create summary gif from user screenshots"""
        if not self.current_session:
            return False
            
        session_dir = self.current_session['dir']
        
        # Create animated gif only from user screenshots
        gif_path = self._create_summary_gif(session_dir)
        
        # Save session metadata and events
        metadata = {
            'session_id': self.current_session['id'],
            'name': self.current_session['name'],
            'start_time': self.current_session['start_time'].isoformat(),
            'end_time': datetime.now().isoformat(),
            'events': [event.to_dict() for event in events],
            'conversation_history': conversation_history,
            'screenshots': self.current_session['screenshots'],
            'summary_gif': gif_path
        }
        
        with open(os.path.join(session_dir, 'session.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return True
        
    def _create_summary_gif(self, session_dir):
        """Create animated gif from user screenshots"""
        screenshots = self.current_session['screenshots']
        if not screenshots:
            return None
            
        # Sort screenshots by timestamp
        screenshots.sort(key=lambda x: x['timestamp'])
        
        # Load and resize images
        images = []
        thumbnail_size = (320, 180)  # 16:9 aspect ratio
        
        for screenshot in screenshots:
            try:
                with Image.open(screenshot['path']) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Create copy of image before thumbnail to avoid modifying original
                    thumb = img.copy()
                    thumb.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                    images.append(thumb)
            except Exception as e:
                self.logger.error(f"Error processing screenshot {screenshot['path']}: {e}")
                continue
                
        if images:
            gif_path = os.path.join(session_dir, 'user_screenshots.gif')
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=1000,  # 1 second between frames
                loop=0
            )
            return gif_path
        return None
            
class WorkflowSession:
    def __init__(self):
        # Generate a unique session ID using UUID4 instead of hashlib for better uniqueness
        self.session_id = uuid.uuid4().hex
        
        # Track session start time for duration calculations
        self.start_time = datetime.now()
        
        # Lists to store workflow data
        self.events: List[WorkflowEvent] = []  # All workflow events
        self.ai_interactions: List[AIInteraction] = []  # AI-specific interactions
        self.context_switches: List[tuple[WindowContext, WindowContext]] = []  # Window switches
        
        # Track current active window
        self.current_window: Optional[WindowContext] = None
        
        # Session metadata including system info and active apps
        self.metadata: Dict = {
            'system_info': self._get_system_info(),
            'active_applications': set()  # Using set for unique apps
        }
    
    def _get_system_info(self) -> Dict:
        """
        Collect system information for session context
        Returns dict with platform details and hardware info
        """
        return {
            'platform': platform.system(),  # OS name (Windows, Linux, etc)
            'platform_release': platform.release(),  # OS version
            'cpu_count': psutil.cpu_count(),  # Number of CPU cores
            'memory_total': psutil.virtual_memory().total  # Total RAM in bytes
        }
    
    def add_event(self, event: WorkflowEvent) -> None:
        self.events.append(event)
        
        # Track AI interactions
        if event.ai_context:
            self.ai_interactions.append(event.ai_context)
        
        # Track context switches
        if self.current_window and event.window_context.app_name != self.current_window.app_name:
            self.context_switches.append((self.current_window, event.window_context))
        
        # Update current window
        self.current_window = event.window_context
        
        # Track active applications
        self.metadata['active_applications'].add(event.window_context.app_name)
    
    def export_session(self, filepath: str) -> None:
        session_data = {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'events': [event.to_dict() for event in self.events],
            'ai_interactions': len(self.ai_interactions),
            'context_switches': len(self.context_switches),
            'metadata': {
                **self.metadata,
                'active_applications': list(self.metadata['active_applications'])
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)

class ScreenshotManager:
    def __init__(self):
        self.screenshot_dir = "screenshots"
        self.max_screenshots = 100  # Limit total screenshots
        self._cleanup_threshold = 90  # Clean up when reaching this many
        self.logger = logging.getLogger('WorkflowAnalyzer.Screenshots')
        
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
            
    def capture_screenshot(self) -> str:
        """Capture screenshot with resource management"""
        try:
            # Check screenshot count and cleanup if needed
            screenshots = os.listdir(self.screenshot_dir)
            if len(screenshots) >= self._cleanup_threshold:
                self._cleanup_old_screenshots()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.screenshot_dir}/screenshot_{timestamp}.png"
            
            # Capture at reduced size to save memory
            screenshot = pyautogui.screenshot()
            # Resize to half size while maintaining aspect ratio
            width, height = screenshot.size
            screenshot = screenshot.resize((width//2, height//2), Image.Resampling.LANCZOS)
            screenshot.save(filename, optimize=True, quality=85)  # Compress to save space
            
            return filename
        except Exception as e:
            self.logger.error(f"Screenshot capture failed: {e}")
            return None
            
    def _cleanup_old_screenshots(self):
        """Remove oldest screenshots when limit is reached"""
        try:
            screenshots = [(f, os.path.getctime(os.path.join(self.screenshot_dir, f))) 
                         for f in os.listdir(self.screenshot_dir)]
            screenshots.sort(key=lambda x: x[1])  # Sort by creation time
            
            # Remove oldest files until below threshold
            while len(screenshots) >= self._cleanup_threshold:
                oldest_file = screenshots.pop(0)[0]
                os.remove(os.path.join(self.screenshot_dir, oldest_file))
        except Exception as e:
            self.logger.error(f"Screenshot cleanup failed: {e}")
class TaskTracker:
    def __init__(self):
        self.current_task: Optional[TaskContext] = None
        self.task_history: List[TaskContext] = []
        self.ai_interaction_start: Optional[datetime] = None
        self.current_interaction_type: Optional[InteractionType] = None
        
    def start_task(self, category: TaskCategory, details: Optional[Dict] = None) -> None:
        """Start a new task"""
        self.current_task = TaskContext(category, details)
        self.task_history.append(self.current_task)
        
    def start_ai_interaction(self, interaction_type: InteractionType) -> None:
        """Mark the start of an AI interaction"""
        self.ai_interaction_start = datetime.now()
        self.current_interaction_type = interaction_type
        self.start_task(TaskCategory.AI_INTERACTION, {
            'interaction_type': interaction_type.name,
            'start_time': self.ai_interaction_start.isoformat()
        })
        
    def end_ai_interaction(self) -> None:
        """Mark the end of an AI interaction"""
        if self.ai_interaction_start:
            duration = datetime.now() - self.ai_interaction_start
            if self.current_task and self.current_task.category == TaskCategory.AI_INTERACTION:
                self.current_task.details['duration'] = str(duration)
                self.current_task.details['end_time'] = datetime.now().isoformat()
            self.ai_interaction_start = None
            self.current_interaction_type = None


class WorkflowTracker:
    def __init__(self, async_handler: AsyncWorkflowHandler):
        self.async_handler = async_handler
        self.events = deque(maxlen=1000)
        self.mouse_listener = None
        self.keyboard_listener = None
        self.logger = logging.getLogger('WorkflowAnalyzer.Tracker')
        self._running = False
        
    def _on_click(self, x: int, y: int, button, pressed: bool) -> None:
        """Handle mouse click events"""
        if not pressed:  # Only track release events
            try:
                event = WorkflowEvent(
                    timestamp=datetime.now(),
                    event_type='mouse_click',
                    window_title=self._get_window_title(),
                    details={
                        'position': (x, y),
                        'button': str(button)
                    }
                )
                
                self.events.append(event)
                if self._running:
                    self.async_handler.loop.call_soon_threadsafe(
                        lambda: self.async_handler.event_queue.put_nowait(event)
                    )
            except Exception as e:
                self.logger.error(f"Click handler error: {e}")
                
    def _on_key(self, key) -> None:
        """Handle keyboard events (F1-F12 keys only)"""
        try:
            # Only track function keys (F1-F12)
            if hasattr(key, 'vk') and 111 < key.vk < 124:
                event = WorkflowEvent(
                    timestamp=datetime.now(),
                    event_type='key_press',
                    window_title=self._get_window_title(),
                    details={
                        'key': str(key)
                    }
                )
                
                self.events.append(event)
                if self._running:
                    self.async_handler.loop.call_soon_threadsafe(
                        lambda: self.async_handler.event_queue.put_nowait(event)
                    )
        except Exception as e:
            self.logger.error(f"Key handler error: {e}")
            
    def _get_window_title(self) -> str:
        """Get the current active window title"""
        try:
            hwnd = win32gui.GetForegroundWindow()
            return win32gui.GetWindowText(hwnd)
        except Exception:
            return "Unknown Window"
            
    def start(self):
        """Start tracking with basic error handling"""
        if self._running:
            return
            
        try:
            self._running = True
            self.mouse_listener = mouse.Listener(
                on_click=self._on_click,
                on_move=None,    # Disable move events
                on_scroll=None   # Disable scroll events
            )
            self.keyboard_listener = keyboard.Listener(
                on_press=self._on_key,
                on_release=None  # Disable release events
            )
            
            self.mouse_listener.start()
            self.keyboard_listener.start()
        except Exception as e:
            self.logger.error(f"Start error: {e}")
            self.stop()
            
    def stop(self):
        """Stop tracking and clean up listeners"""
        self._running = False
        
        # Stop mouse listener
        if self.mouse_listener:
            try:
                self.mouse_listener.stop()
            except Exception as e:
                self.logger.error(f"Error stopping mouse listener: {e}")
            finally:
                self.mouse_listener = None
                
        # Stop keyboard listener
        if self.keyboard_listener:
            try:
                self.keyboard_listener.stop()
            except Exception as e:
                self.logger.error(f"Error stopping keyboard listener: {e}")
            finally:
                self.keyboard_listener = None
                
        # Clear any remaining events
        self.events.clear()

class AsyncWorkflowHandler:
    def __init__(self, api_key: str, loop: asyncio.AbstractEventLoop):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.event_queue = asyncio.Queue(maxsize=1000)
        self.analysis_queue = asyncio.Queue(maxsize=10)
        self.running = False
        self.logger = logging.getLogger('WorkflowAnalyzer.AsyncHandler')
        self.loop = loop
        self._tasks = []
    
    
    

    async def _process_events(self):
        """Process events from queue"""
        while self.running:
            try:
                event = await self.event_queue.get()
                if event and hasattr(self, 'ui_callback'):
                    update = {
                        'type': 'event',
                        'data': {
                            'timestamp': event.timestamp.strftime('%H:%M:%S'),
                            'event_type': event.event_type,
                            'window_title': event.window_title,
                            'details': event.details
                        }
                    }
                    await self.ui_callback(update)
                self.event_queue.task_done()
            except Exception as e:
                self.logger.error(f"Event processing error: {e}")
                await asyncio.sleep(0.1)

    async def _process_analysis(self):
        """Process analysis requests"""
        while self.running:
            try:
                request = await self.analysis_queue.get()
                if request:
                    result = await self._perform_analysis(request)
                    
                    if hasattr(self, 'ui_callback'):
                        await self.ui_callback({
                            'type': 'analysis',
                            'data': result['content'] if result['success'] else f"Analysis failed: {result['error']}"
                        })
                self.analysis_queue.task_done()
            except Exception as e:
                self.logger.error(f"Analysis processing error: {e}")
                await asyncio.sleep(0.1)

    async def start(self):
        """Start all async tasks"""
        if not self.running:
            self.running = True
            self._tasks = [
                self.loop.create_task(self._process_events()),
                self.loop.create_task(self._process_analysis())
            ]
            self.logger.debug("AsyncWorkflowHandler started")

    async def stop(self):
        """Stop all async tasks"""
        self.logger.debug("Stopping AsyncWorkflowHandler")
        self.running = False
        
        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                
        if self._tasks:
            # Wait for tasks to complete/cancel with timeout
            try:
                await asyncio.wait(self._tasks, timeout=2.0)
            except Exception as e:
                self.logger.error(f"Error waiting for tasks to complete: {e}")
            finally:
                self._tasks.clear()

    
    
    def __init__(self, api_key: str, loop: asyncio.AbstractEventLoop):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.event_queue = asyncio.Queue(maxsize=1000)
        self.analysis_queue = asyncio.Queue(maxsize=10)
        self.running = False
        self.logger = logging.getLogger('WorkflowAnalyzer.AsyncHandler')
        self.loop = loop
        self._tasks = []

    async def _process_analysis(self):
        """Process analysis requests"""
        while self.running:
            try:
                request = await self.analysis_queue.get()
                if request:
                    result = await self._perform_analysis(request)
                    
                    if hasattr(self, 'ui_callback'):
                        await self.ui_callback({
                            'type': 'analysis',
                            'data': result['content'] if result['success'] else f"Analysis failed: {result['error']}"
                        })
                self.analysis_queue.task_done()
            except Exception as e:
                self.logger.error(f"Analysis processing error: {e}")
                await asyncio.sleep(0.1)

    async def _perform_analysis(self, request):
        """Perform analysis using Claude API"""
        try:
            messages = [{
                "role": "user", 
                "content": []
            }]

            # Add text context
            messages[0]["content"].append({
                "type": "text",
                "text": request['context']
            })

            # Add screenshot if present
            if 'screenshot' in request and request['screenshot']:
                try:
                    # Convert PIL Image to bytes using a fresh BytesIO buffer
                    img_byte_arr = io.BytesIO()
                    request['screenshot'].save(img_byte_arr, format='PNG', optimize=True)
                    img_byte_arr.seek(0)  # Reset buffer position to start
                    img_bytes = img_byte_arr.getvalue()
                    
                    # Convert to base64 with padding
                    base64_image = base64.b64encode(img_bytes).decode('utf-8')
                    
                    # Add image to message content with correct format
                    messages[0]["content"].append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image
                        }
                    })
                    
                    # Log success but not the actual image data
                    self.logger.debug("Successfully encoded image for API request")
                    
                except Exception as img_error:
                    self.logger.error(f"Image processing error: {img_error}")
                    raise Exception(f"Failed to process image: {img_error}")

            # Log the request structure (without the actual image data)
            debug_messages = copy.deepcopy(messages)
            if len(debug_messages[0]["content"]) > 1:
                debug_messages[0]["content"][1]["source"]["data"] = "<image data>"
            self.logger.debug(f"Sending request with messages: {debug_messages}")

            # Make API call with error handling
            try:
                response = await self.loop.run_in_executor(
                    None,
                    lambda: self.client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1024,
                        messages=messages
                    )
                )
                
                if response.content:
                    content = response.content[0].text if isinstance(response.content, list) else response.content
                    return {
                        'success': True,
                        'content': content,
                        'request_id': request.get('id')
                    }
                else:
                    return {
                        'success': False,
                        'error': "No response content",
                        'request_id': request.get('id')
                    }
                    
            except Exception as api_error:
                error_msg = str(api_error)
                if hasattr(api_error, 'response'):
                    try:
                        error_msg = f"Error code: {api_error.response.status_code} - {api_error.response.json()}"
                    except:
                        pass
                self.logger.error(f"API call error: {error_msg}")
                raise Exception(f"API call failed: {error_msg}") 
        except Exception as e:
            self.logger.error(f"Analysis error: {e}")
            return {
                'success': False,
                'error': str(e),
                'request_id': request.get('id')
            }

    

    def _sync_perform_analysis(self, request):
        """Synchronous version of perform_analysis for use with run_in_executor"""
        return asyncio.run(self._perform_analysis(request))
        
    async def queue_message_for_analysis(self, message, screenshot=None):
        """Queue a message for analysis with optional screenshot"""
        try:
            # Create base request
            request = {
                'id': str(uuid.uuid4()),
                'context': message,
        }
        
            # Add screenshot if provided
            if screenshot:
                request['screenshot'] = screenshot
                self.logger.debug("Screenshot included in analysis request")
            else:
                self.logger.debug("No screenshot included in analysis request")
            
        # Queue the request
            await self.analysis_queue.put(request)
            return True
        except Exception as e:
            self.logger.error(f"Error queueing message: {e}")
            return False

def send_message(self):
    """Send user message and trigger analysis"""
    message = self.message_input.get().strip()
    if message:
        # Add message to conversation history
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.conversation_history.append({
            'role': 'user',
            'content': message,
            'timestamp': timestamp
        })
        
        # Update conversation display
        self.conversation_text.insert(
            tk.END,
            f"\n[{timestamp}] You: {message}\n"
        )
        self.conversation_text.see(tk.END)
        
        # Clear input
        self.message_input.delete(0, tk.END)
        
        # Queue message for analysis
        async def process():
            return await self.async_handler.queue_message_for_analysis(
                f"\nUser Message: {message}\n\n"
                "Please analyze this message in the context of the current workflow session."
            )
        
        self._schedule_async_task(process())

def _take_screenshot(self):
    """Handle user-initiated screenshot capture"""
    if not self.recording:
        messagebox.showwarning(
            "Warning", 
            "Please start recording a session before taking screenshots"
        )
        return
        
    if not hasattr(self, 'screenshot_selector'):
        self.screenshot_selector = ScreenshotSelector(self.root)
        
    screenshot = self.screenshot_selector.start_selection()
    if screenshot:
        try:
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"user_screenshot_{timestamp}.png"
            filepath = os.path.join(
                self.session_manager.current_session['dir'],
                "screenshots",
                filename
            )
            screenshot.save(filepath)
            
            # Add to session screenshots
            self.session_manager.current_session['screenshots'].append({
                'path': filepath,
                'timestamp': timestamp,
                'description': 'User-selected area for analysis'
            })
            
            # Update conversation display
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.conversation_history.append({
                'role': 'user',
                'content': '[Screenshot analysis request]',
                'timestamp': timestamp,
                'screenshot_path': filepath
            })
            
            self.conversation_text.insert(
                tk.END,
                f"\n[{timestamp}] You: Requested analysis of captured screenshot\n"
            )
            self.conversation_text.see(tk.END)
            
            # Queue screenshot for analysis
            async def process():
                return await self.async_handler.queue_message_for_analysis(
                    "Please analyze this screenshot that I've just captured. "
                    "Describe what you see in detail and provide any relevant insights "
                    "about the content shown.",
                    screenshot
                )
            
            self._schedule_async_task(process())
            self.logger.debug("Screenshot captured and queued for analysis")
            
        except Exception as e:
            self.logger.error(f"Screenshot analysis error: {e}")
            messagebox.showerror(
                "Error",
                f"Failed to process screenshot: {str(e)}"
            )
            
class WorkflowAnalyzerUI:
    def __init__(self, api_key: str):
        self.root = tk.Tk()
        self.root.title("Workflow Analyzer")
        self.logger = logging.getLogger('WorkflowAnalyzer.UI')
        self.conversation_history = []
        self.recording = False
        
        # Set up async event loop first
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Initialize async handler with the loop
        self.async_handler = AsyncWorkflowHandler(api_key, self.loop)
        
        # Initialize session manager
        self.session_manager = SessionManager()
        
        # Initialize tracker
        self.tracker = WorkflowTracker(self.async_handler)
        
        # Set up UI components
        self.setup_ui()  # Renamed to match the method definition
        self._setup_async_handling()
    def save_session(self):
        """Explicitly save the current session"""
        try:
            if not self.session_manager.current_session:
                messagebox.showwarning(
                    "Warning",
                    "No active session to save. Please start recording first."
                )
                return
                
            # Save session data
            if self.session_manager.save_session(
                list(self.tracker.events),
                self.conversation_history
            ):
                messagebox.showinfo(
                    "Success",
                    "Session saved successfully!"
                )
                
                # Show summary GIF location if created
                if self.session_manager.current_session.get('summary_gif'):
                    self.conversation_text.insert(
                        tk.END,
                        f"\n[System] Session summary GIF created: "
                        f"{self.session_manager.current_session['summary_gif']}\n"
                    )
                    self.conversation_text.see(tk.END)
            else:
                messagebox.showerror(
                    "Error",
                    "Failed to save session"
                )
        except Exception as e:
            self.logger.error(f"Error saving session: {e}")
            messagebox.showerror(
                "Error",
                f"Failed to save session: {str(e)}"
            )
    def _setup_async_handling(self):
        """Set up async event handling"""
        def run_async_loop():
            """Run the async event loop in a separate thread"""
            asyncio.set_event_loop(self.loop)
            try:
                self.loop.run_forever()
            except Exception as e:
                self.logger.error(f"Event loop error: {e}")
            finally:
                try:
                    # Cancel all running tasks
                    pending = asyncio.all_tasks(self.loop)
                    for task in pending:
                        task.cancel()
                    # Wait for tasks to complete/cancel
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                finally:
                    self.loop.close()
            
        # Start async loop in separate thread
        self.async_thread = threading.Thread(target=run_async_loop, daemon=True)
        self.async_thread.start()
        
        # Set up UI callback
        async def ui_callback(update):
            """Process updates in the UI thread"""
            self.root.after(0, lambda: self._process_update(update))
        
        self.async_handler.ui_callback = ui_callback

    def _process_update(self, update):
        """Process updates in the UI thread"""
        try:
            if update['type'] == 'event':
                self._update_event_list(update['data'])
            elif update['type'] == 'analysis':
                # Add Claude's response to conversation history
                timestamp = datetime.now().strftime('%H:%M:%S')
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': update['data'],
                    'timestamp': timestamp
                })
                
                # Update both conversation and analysis displays
                self.conversation_text.insert(
                    tk.END,
                    f"\n[{timestamp}] Claude: {update['data']}\n"
                )
                self.conversation_text.see(tk.END)
                self._update_analysis_text(update['data'])
            elif update['type'] == 'error':
                self._show_error(update['data'])
        except Exception as e:
            self.logger.error(f"Error processing update: {e}")

    def _update_event_list(self, event_data):
        """Update event list in UI"""
        try:
            self.event_tree.insert('', 0, values=(
                event_data['timestamp'],
                event_data['event_type'],
                event_data['window_title'],
                json.dumps(event_data['details'])
            ))
            
            # Maintain reasonable number of visible items
            items = self.event_tree.get_children()
            if len(items) > 100:  # Keep last 100 events visible
                self.event_tree.delete(items[-1])
        except Exception as e:
            self.logger.error(f"Error updating event list: {e}")

    def _update_analysis_text(self, text):
        """Update analysis text in UI"""
        try:
            self.analysis_text.delete('1.0', tk.END)
            self.analysis_text.insert('1.0', text)
        except Exception as e:
            self.logger.error(f"Error updating analysis text: {e}")

    def _show_error(self, error_text):
        """Show error in UI"""
        try:
            messagebox.showerror("Error", str(error_text))
        except Exception as e:
            self.logger.error(f"Error showing error dialog: {e}")

    def run(self):
        """Start the application"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.root.mainloop()
        finally:
            # Clean up async resources
            if self.recording:
                self.recording = False
                self._schedule_async_task(self.async_handler.stop())
                self.tracker.stop()
            
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            
            if hasattr(self, 'async_thread'):
                self.async_thread.join(timeout=1.0)
        
    def setup_ui(self):  # Changed from _setup_ui to setup_ui
        """Set up the UI components"""
        # Create main frame with paned window for resizable sections
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add session frame
        session_frame = ttk.LabelFrame(main_frame, text="Session")
        session_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(session_frame, text="Session Name:").pack(side=tk.LEFT, padx=5)
        self.session_name = ttk.Entry(session_frame)
        self.session_name.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Create paned window for main sections
        paned = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Events section
        event_frame = ttk.LabelFrame(paned, text="Events")
        paned.add(event_frame)
        
        self.event_tree = ttk.Treeview(
            event_frame,
            columns=('time', 'type', 'window', 'details'),
            show='headings'
        )
        
        self.event_tree.heading('time', text='Time')
        self.event_tree.heading('type', text='Type')
        self.event_tree.heading('window', text='Window')
        self.event_tree.heading('details', text='Details')
        
        event_scroll = ttk.Scrollbar(event_frame, orient=tk.VERTICAL, command=self.event_tree.yview)
        self.event_tree.configure(yscrollcommand=event_scroll.set)
        
        self.event_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        event_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Conversation section
        conv_frame = ttk.LabelFrame(paned, text="Conversation")
        paned.add(conv_frame)
        
        self.conversation_text = scrolledtext.ScrolledText(conv_frame, height=10)
        self.conversation_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        input_frame = ttk.Frame(conv_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.message_input = ttk.Entry(input_frame)
        self.message_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.message_input.bind('<Return>', lambda e: self.send_message())
        
        ttk.Button(input_frame, text="Send", command=self.send_message).pack(side=tk.RIGHT, padx=5)
        
        # Analysis section
        analysis_frame = ttk.LabelFrame(paned, text="Analysis")
        paned.add(analysis_frame)
        
        self.analysis_text = scrolledtext.ScrolledText(analysis_frame, height=10)
        self.analysis_text.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Left-aligned buttons
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.start_button = ttk.Button(left_buttons, text="Start Recording", command=self.start_recording)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(left_buttons, text="Stop Recording",
                                    command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(left_buttons, text="Save Session", 
                  command=self.save_session).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(left_buttons, text="Analyze", 
                  command=self.trigger_analysis).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(left_buttons, text="Clear Conversation", 
                  command=self.clear_conversation).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(left_buttons, text="Capture Screenshot", 
                  command=self._take_screenshot).pack(side=tk.LEFT, padx=5)

        # Add window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _take_screenshot(self):
        """Handle user-initiated screenshot capture"""
        if not self.recording:
            messagebox.showwarning(
                "Warning", 
                "Please start recording a session before taking screenshots"
            )
            return
            
        if not hasattr(self, 'screenshot_selector'):
            self.screenshot_selector = ScreenshotSelector(self.root)
            
        screenshot = self.screenshot_selector.start_selection()
        if screenshot:
            try:
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"user_screenshot_{timestamp}.png"
                filepath = os.path.join(
                    self.session_manager.current_session['dir'],
                    "screenshots",
                    filename
                )
                
                # Ensure the screenshot is in RGB mode
                if screenshot.mode != 'RGB':
                    screenshot = screenshot.convert('RGB')
                    
                # Save with specific format
                screenshot.save(filepath, format='PNG', optimize=True)
                
                # Add to session screenshots
                self.session_manager.current_session['screenshots'].append({
                    'path': filepath,
                    'timestamp': timestamp,
                    'description': 'User-selected area for analysis'
                })
                
                # Update conversation display
                timestamp = datetime.now().strftime('%H:%M:%S')
                self.conversation_history.append({
                    'role': 'user',
                    'content': '[Screenshot analysis request]',
                    'timestamp': timestamp,
                    'screenshot_path': filepath
                })
                
                self.conversation_text.insert(
                    tk.END,
                    f"\n[{timestamp}] You: Requested analysis of captured screenshot\n"
                )
                self.conversation_text.see(tk.END)
                
                # Queue screenshot for analysis
                async def process():
                    return await self.async_handler.queue_message_for_analysis(
                        "Please analyze this screenshot that I've just captured. "
                        "Describe what you see in detail and provide any relevant insights "
                        "about the content shown.",
                        screenshot
                    )
                
                self._schedule_async_task(process())
                self.logger.debug("Screenshot captured and queued for analysis")
                
            except Exception as e:
                self.logger.error(f"Screenshot analysis error: {e}")
                messagebox.showerror(
                    "Error",
                    f"Failed to process screenshot: {str(e)}"
                )

    def send_message(self):
        """Send user message and trigger analysis"""
        message = self.message_input.get().strip()
        if message:
            # Add message to conversation history
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.conversation_history.append({
                'role': 'user',
                'content': message,
                'timestamp': timestamp
            })
            
            # Update conversation display
            self.conversation_text.insert(
                tk.END,
                f"\n[{timestamp}] You: {message}\n"
            )
            self.conversation_text.see(tk.END)
            
            # Clear input
            self.message_input.delete(0, tk.END)
            
            # Prepare analysis request
            request = {
                'id': str(uuid.uuid4()),
                'context': (
                    f"\nUser Message: {message}\n\n"
                    "Please analyze this message in the context of the current workflow session."
                )
            }
            
            # Queue for analysis using call_soon_threadsafe
            self._schedule_async_task(
                lambda: self.async_handler.analysis_queue.put_nowait(request)
            )

    def clear_conversation(self):
        """Clear conversation history and display"""
        self.conversation_history.clear()
        self.conversation_text.delete('1.0', tk.END)
        self.analysis_text.delete('1.0', tk.END)

    def trigger_analysis(self):
        """Trigger analysis of current workflow session"""
        if self.recording:
            events = list(self.tracker.events)
            
            # Prepare context for analysis
            context = self._prepare_analysis_context(events)
            
            # Create analysis request
            request = {
                'id': str(uuid.uuid4()),
                'context': context
            }
            
            # Queue for analysis
            self._schedule_async_task(
                self.async_handler.analysis_queue.put_nowait(request)
            )
        else:
            self.conversation_text.insert(
                tk.END,
                "\n[System] Please start recording to enable analysis\n"
            )
            self.conversation_text.see(tk.END)

    def _prepare_analysis_context(self, events):
        """Prepare context for analysis request"""
        template = """
        Analyzing workflow session with focus on AI interactions:
        
        Session Duration: {start_time} to {end_time}
        Total Events: {event_count}
        
        AI Platform Interactions:
        {ai_interactions}
        
        Context Switches:
        {context_switches}
        
        Recent Conversation History:
        {conversation}
        
        Please analyze this workflow context and provide insights on:
        1. Current workflow patterns and efficiency
        2. User interaction style and preferences
        3. Potential optimizations or suggestions
        """
        
        if not events:
            return template.format(
                start_time="N/A",
                end_time="N/A",
                event_count=0,
                ai_interactions="No AI interactions recorded",
                context_switches="No context switches recorded",
                conversation=self._format_conversation()
            )
            
        # Get time range
        start_time = min(e.timestamp for e in events)
        end_time = max(e.timestamp for e in events)
        
        # Extract AI interactions and context switches
        ai_interactions = []
        context_switches = []
        prev_window = None
        
        for event in events:
            # Look for AI platform windows
            if any(ai_term in event.window_title.lower() 
                  for ai_term in ['chat', 'gpt', 'claude', 'anthropic']):
                ai_interactions.append(
                    f"[{event.timestamp.strftime('%H:%M:%S')}] "
                    f"{event.window_title}: {event.event_type}"
                )
            
            # Track context switches
            if prev_window and prev_window != event.window_title:
                context_switches.append(
                    f"[{event.timestamp.strftime('%H:%M:%S')}] "
                    f"{prev_window} -> {event.window_title}"
                )
            prev_window = event.window_title
            
        return template.format(
            start_time=start_time.strftime("%H:%M:%S"),
            end_time=end_time.strftime("%H:%M:%S"),
            event_count=len(events),
            ai_interactions="\n".join(ai_interactions) or "No AI interactions detected",
            context_switches="\n".join(context_switches[-5:]) or "No context switches recorded",
            conversation=self._format_conversation()
        )

    def _format_conversation(self):
        """Format recent conversation history"""
        if not self.conversation_history:
            return "No conversation history"
            
        # Get last 5 messages
        recent_msgs = self.conversation_history[-5:]
        return "\n".join(
            f"[{msg['timestamp']}] {msg['role'].title()}: {msg['content']}"
            for msg in recent_msgs
        )

    def _schedule_async_task(self, task):
        """Schedule a coroutine to run in the async loop"""
        if self.loop and self.loop.is_running():
            if asyncio.iscoroutine(task):
                asyncio.run_coroutine_threadsafe(task, self.loop)
            else:
                self.loop.call_soon_threadsafe(task)

    def start_recording(self):
        """Start recording workflow events"""
        name = self.session_name.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a session name")
            return
            
        if not self.recording:
            self.session_manager.create_session(name)
            self.recording = True
            
            # Schedule async start
            async def start_async():
                await self.async_handler.start()
            
            self._schedule_async_task(start_async())
            self.tracker.start()
            
            # Update UI
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            self.session_name.configure(state=tk.DISABLED)
            self.conversation_text.insert(
                tk.END,
                f"\n[System] Recording started - Session: {name}\n"
            )
            self.conversation_text.see(tk.END)

    def stop_recording(self):
        """Stop recording and save session"""
        if self.recording:
            self.recording = False
            
            # Schedule async stop
            async def stop_async():
                await self.async_handler.stop()
            
            self._schedule_async_task(stop_async())
            self.tracker.stop()
            
            try:
                # Save session
                if self.session_manager.save_session(
                    list(self.tracker.events),
                    self.conversation_history
                ):
                    if hasattr(self, 'start_button') and self.start_button.winfo_exists():
                        self.start_button.configure(state=tk.NORMAL)
                    if hasattr(self, 'stop_button') and self.stop_button.winfo_exists():
                        self.stop_button.configure(state=tk.DISABLED)
                    if hasattr(self, 'session_name') and self.session_name.winfo_exists():
                        self.session_name.configure(state=tk.NORMAL)
                    
                    self.conversation_text.insert(
                        tk.END,
                        "\n[System] Recording stopped and session saved\n"
                    )
                    
                    if self.session_manager.current_session.get('summary_gif'):
                        self.conversation_text.insert(
                            tk.END,
                            f"\n[System] Session summary GIF created: "
                            f"{self.session_manager.current_session['summary_gif']}\n"
                        )
                else:
                    self.conversation_text.insert(
                        tk.END,
                        "\n[System] Error saving session\n"
                    )
                self.conversation_text.see(tk.END)
            except Exception as e:
                self.logger.error(f"Error in stop_recording: {e}")
    def _on_closing(self):
        """Handle window closing event"""
        try:
            if self.recording:
                if messagebox.askokcancel("Quit", "Recording is still active. Do you want to stop recording and quit?"):
                    self.stop_recording()
                else:
                    return
                    
            # Create an async task to stop the handler
            async def cleanup():
                if hasattr(self.async_handler, 'stop'):
                    await self.async_handler.stop()
            
            # Run cleanup
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(cleanup(), self.loop)
                
            self.root.destroy()
        except Exception as e:
            self.logger.error(f"Error in _on_closing: {e}")
            self.root.destroy()

def main():
    logger = setup_logging()
    try:
        API_KEY = "api_key"
        app = WorkflowAnalyzerUI(API_KEY)
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
    finally:
        logging.shutdown()

if __name__ == "__main__":
    main()