from __future__ import annotations  

from tkinter import messagebox
import win32process
import psutil
import asyncio
import win32gui
import win32con
from pynput import mouse, keyboard
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import anthropic
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import Optional, Dict, List
import json
from queue import Queue, Empty
import threading
import sys
import logging
import logging.handlers
import codecs
import pyautogui
from PIL import Image
import io
import base64
import os
import shutil
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Type
    from asyncio import AbstractEventLoop

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
class WorkflowEvent:
    timestamp: datetime
    event_type: str
    window_title: str
    window_class: str
    process_name: str
    details: Dict

class ScreenshotManager:
    def __init__(self):
        self.screenshot_dir = "screenshots"
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
            
    def capture_screenshot(self) -> str:
        """Capture screenshot and return filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.screenshot_dir}/screenshot_{timestamp}.png"
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        return filename

class WorkflowTracker:
    def __init__(self, async_handler: AsyncWorkflowHandler):
        self.async_handler = async_handler
        self.events = deque(maxlen=1000)
        self.mouse_listener = mouse.Listener(on_click=self._on_click)
        self.keyboard_listener = keyboard.Listener(on_press=self._on_key)
        self.screenshot_manager = ScreenshotManager()
        self.current_window = None
        self.last_window_change = datetime.now()
        self.logger = logging.getLogger('WorkflowAnalyzer.Tracker')

    def queue_event(self, event: WorkflowEvent):
        """Queue event for async processing - non-async version"""
        try:
            if self.async_handler.loop and self.async_handler.loop.is_running():
                self.events.append(event)
                self.async_handler.loop.call_soon_threadsafe(
                    lambda: self.async_handler.event_queue.put_nowait(event)
                )
        except Exception as e:
            self.logger.error(f"Error queuing event: {e}")
        
    def _on_click(self, x, y, button, pressed):
        if not pressed:
            try:
                window_info = self._get_window_info()
                screenshot_event = self._check_window_change()
                
                event = WorkflowEvent(
                    timestamp=datetime.now(),
                    event_type='mouse_click',
                    window_title=window_info['title'],
                    window_class=window_info['class'],
                    process_name=window_info['process'],
                    details={
                        'position': (x, y),
                        'button': str(button),
                        'screenshot_context': screenshot_event
                    }
                )
                
                self.queue_event(event)
                
            except Exception as e:
                self.logger.error(f"Error in mouse click handler: {e}")
            
    def _on_key(self, key):
        if hasattr(key, 'vk') and key.vk in range(112, 124):
            try:
                window_info = self._get_window_info()
                screenshot_event = self._check_window_change()
                
                event = WorkflowEvent(
                    timestamp=datetime.now(),
                    event_type='key_press',
                    window_title=window_info['title'],
                    window_class=window_info['class'],
                    process_name=window_info['process'],
                    details={
                        'key': str(key),
                        'screenshot_context': screenshot_event
                    }
                )
                
                self.queue_event(event)
                
            except Exception as e:
                self.logger.error(f"Error in key press handler: {e}")

    def _get_window_info(self) -> Dict:
        """Get information about current active window"""
        hwnd = win32gui.GetForegroundWindow()
        return {
            'title': win32gui.GetWindowText(hwnd),
            'class': win32gui.GetClassName(hwnd),
            'process': 'Unknown'  # Would need additional logic to get process name
        }

    def _check_window_change(self) -> Optional[str]:
        """Check if window changed and capture screenshot if needed"""
        current_window = win32gui.GetForegroundWindow()
        if current_window != self.current_window:
            self.current_window = current_window
            self.last_window_change = datetime.now()
            return self.screenshot_manager.capture_screenshot()
        return None

    def start(self):
        """Start tracking events"""
        self.mouse_listener.start()
        self.keyboard_listener.start()

    def stop(self):
        """Stop tracking events"""
        self.mouse_listener.stop()
        self.keyboard_listener.stop()

class AsyncWorkflowHandler:
    """Manages async operations between UI and analysis components"""
    def __init__(self, api_key: str, loop: asyncio.AbstractEventLoop):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.event_queue = asyncio.Queue()
        self.analysis_queue = asyncio.Queue()
        self.running = False
        self.logger = logging.getLogger('WorkflowAnalyzer.AsyncHandler')
        self.loop = loop
        
    def _prepare_ui_update(self, event):
        """Prepare event data for UI update"""
        return {
            'type': 'event',
            'data': {
                'timestamp': event.timestamp.strftime('%H:%M:%S'),
                'event_type': event.event_type,
                'window_title': event.window_title,
                'details': event.details
            }
        }
        
    async def _handle_analysis_result(self, result):
        """Handle analysis result and update UI"""
        if result['success']:
            await self._update_ui({
                'type': 'analysis',
                'data': result['content']
            })
        else:
            await self._update_ui({
                'type': 'error',
                'data': f"Analysis failed: {result['error']}"
            })
            
    async def start(self):
        """Start all async tasks"""
        self.running = True
        self.loop.create_task(self._process_events())
        self.loop.create_task(self._process_analysis())
        
    async def stop(self):
        """Gracefully stop all async tasks"""
        self.running = False
        # Wait for queues to empty
        await self.event_queue.join()
        await self.analysis_queue.join()
        
    async def _process_events(self):
        """Process events from queue"""
        while self.running:
            try:
                event = await self.event_queue.get()
                if event:
                    ui_update = self._prepare_ui_update(event)
                    if hasattr(self, 'ui_callback'):
                        await self.ui_callback(ui_update)
                self.event_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error processing event: {str(e)}")
            await asyncio.sleep(0.1)
            
    async def _process_analysis(self):
        """Process analysis requests"""
        while self.running:
            try:
                analysis_request = await self.analysis_queue.get()
                result = await self._perform_analysis(analysis_request)
                await self._handle_analysis_result(result)
                self.analysis_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error processing analysis: {e}")
            await asyncio.sleep(0.1)
            
    async def _handle_event(self, event):
        """Handle single event processing"""
        # Process event and update UI
        ui_update = self._prepare_ui_update(event)
        await self._update_ui(ui_update)
        
    async def _perform_analysis(self, request):
        """Perform analysis using Claude API"""
        try:
            # Create message without await
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": request['context']
                }]
            )
            
            if hasattr(response, 'content') and response.content:
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
        except Exception as e:
            self.logger.error(f"Analysis error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'request_id': request.get('id')
            }        
    async def start(self):
        """Start all async tasks"""
        self.running = True
        self.loop.create_task(self._process_events())
        self.loop.create_task(self._process_analysis())

    async def stop(self):
        """Gracefully stop all async tasks"""
        self.running = False
        await self.event_queue.join()
        await self.analysis_queue.join()

    async def _process_events(self):
        """Process events from queue"""
        while self.running:
            try:
                event = await self.event_queue.get()
                if event:
                    ui_update = self._prepare_ui_update(event)
                    if hasattr(self, 'ui_callback'):
                        await self.ui_callback(ui_update)
                self.event_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error processing event: {str(e)}")
            await asyncio.sleep(0.1)

    async def _process_analysis(self):
        """Process analysis requests"""
        while self.running:
            try:
                analysis_request = await self.analysis_queue.get()
                if analysis_request:
                    result = await self._perform_analysis(analysis_request)
                    await self._handle_analysis_result(result)
                self.analysis_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error processing analysis: {str(e)}")
            await asyncio.sleep(0.1)

    def _prepare_ui_update(self, event):
        """Prepare event data for UI update"""
        return {
            'type': 'event',
            'data': {
                'timestamp': event.timestamp.strftime('%H:%M:%S'),
                'event_type': event.event_type,
                'window_title': event.window_title,
                'details': event.details
            }
        }

    async def _handle_analysis_result(self, result):
        """Handle analysis result and update UI"""
        if result['success']:
            await self._update_ui({
                'type': 'analysis',
                'data': result['content']
            })
        else:
            await self._update_ui({
                'type': 'error',
                'data': f"Analysis failed: {result['error']}"
            })

    async def _update_ui(self, update):
        """Send UI updates through callback"""
        if hasattr(self, 'ui_callback'):
            await self.ui_callback(update)
            
    
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
        
        self._setup_ui()
        self._setup_async_handling()
        
    def _setup_ui(self):
        """Set up the UI components"""
        # Initialize the tracker after async handler is ready
        self.tracker = WorkflowTracker(self.async_handler)
        
        # Create main frame with paned window for resizable sections
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        paned = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Top section for events
        event_frame = ttk.LabelFrame(paned, text="Events")
        paned.add(event_frame)
        
        # Create event tree
        self.event_tree = ttk.Treeview(event_frame, columns=(
            'time', 'type', 'window', 'details'
        ), show='headings')
        
        # Configure columns
        self.event_tree.heading('time', text='Time')
        self.event_tree.heading('type', text='Type')
        self.event_tree.heading('window', text='Window')
        self.event_tree.heading('details', text='Details')
        
        # Add scrollbar for events
        event_scrollbar = ttk.Scrollbar(event_frame, orient=tk.VERTICAL, 
                                      command=self.event_tree.yview)
        self.event_tree.configure(yscrollcommand=event_scrollbar.set)
        
        self.event_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        event_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Middle section for conversation
        conversation_frame = ttk.LabelFrame(paned, text="Conversation")
        paned.add(conversation_frame)
        
        self.conversation_text = scrolledtext.ScrolledText(conversation_frame, height=10)
        self.conversation_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Input frame for user messages
        input_frame = ttk.Frame(conversation_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.message_input = ttk.Entry(input_frame)
        self.message_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.message_input.bind('<Return>', lambda e: self.send_message())
        
        ttk.Button(input_frame, text="Send", 
                   command=self.send_message).pack(side=tk.RIGHT, padx=5)
        
        # Bottom section for analysis
        analysis_frame = ttk.LabelFrame(paned, text="Analysis")
        paned.add(analysis_frame)
        
        self.analysis_text = scrolledtext.ScrolledText(analysis_frame, height=10)
        self.analysis_text.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start Recording", 
                                     command=self.start_recording)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Recording", 
                                    command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Analyze", 
                   command=self.trigger_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Conversation", 
                   command=self.clear_conversation).pack(side=tk.LEFT, padx=5)

    def start_recording(self):
        """Start recording workflow events"""
        if not self.recording:
            self.recording = True
            self._schedule_async_task(self.async_handler.start())
            self.tracker.start()
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            self.conversation_text.insert(tk.END, "\n[System] Recording started\n")
            self.conversation_text.see(tk.END)

    def stop_recording(self):
        """Stop recording workflow events"""
        if self.recording:
            self.recording = False
            self._schedule_async_task(self.async_handler.stop())
            self.tracker.stop()
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self.conversation_text.insert(tk.END, "\n[System] Recording stopped\n")
            self.conversation_text.see(tk.END)

    def send_message(self):
        """Send user message to conversation"""
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
            self.conversation_text.insert(tk.END, f"\n[{timestamp}] You: {message}")
            self.conversation_text.see(tk.END)
            
            # Clear input
            self.message_input.delete(0, tk.END)
            
            # Trigger analysis with updated context
            self.trigger_analysis()

    def clear_conversation(self):
        """Clear conversation history and display"""
        self.conversation_history.clear()
        self.conversation_text.delete('1.0', tk.END)
        self.analysis_text.delete('1.0', tk.END)

    def _prepare_analysis_context(self, events):
        """Prepare context for analysis including conversation history"""
        context_lines = [
            "Here's the current workflow context:",
            "\nRecorded Events:",
        ]
        
        # Add events
        for event in events:
            context_lines.extend([
                f"Time: {event.timestamp}",
                f"Event: {event.event_type}",
                f"Window: {event.window_title}",
                f"Details: {json.dumps(event.details)}",
                ""
            ])
        
        # Add conversation history
        if self.conversation_history:
            context_lines.extend([
                "\nConversation History:",
                ""
            ])
            for msg in self.conversation_history:
                context_lines.append(
                    f"[{msg['timestamp']}] {msg['role'].title()}: {msg['content']}"
                )
        
        # Add request for analysis
        context_lines.extend([
            "",
            "Please analyze this workflow context, including both the recorded events and our conversation.",
            "Provide insights about what the user is doing and respond to any questions or comments in the conversation."
        ])
        
        return "\n".join(context_lines)

    def _update_event_list(self, event_data):
        """Update event list in UI"""
        self.event_tree.insert('', 0, values=(
            event_data['timestamp'],
            event_data['event_type'],
            event_data['window_title'],
            json.dumps(event_data['details'])
        ))
        
    def _update_analysis_text(self, text):
        """Update analysis text in UI"""
        self.analysis_text.delete('1.0', tk.END)
        self.analysis_text.insert('1.0', text)
        
    def _show_error(self, error_text):
        """Show error in UI"""
        messagebox.showerror("Error", error_text)
        
    def _setup_async_handling(self):
        """Set up async event handling"""
        def run_async_loop():
            asyncio.set_event_loop(self.loop)
            try:
                self.loop.run_forever()
            except Exception as e:
                self.logger.error(f"Event loop error: {e}")
            
        # Start async loop in separate thread
        self.async_thread = threading.Thread(target=run_async_loop, daemon=True)
        self.async_thread.start()
        
        # Set up UI callback
        async def ui_callback(update):
            self.root.after(0, lambda: self._process_update(update))
        
        self.async_handler.ui_callback = ui_callback
        
    def _schedule_async_task(self, coro):
        """Schedule a coroutine to run in the async loop"""
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    def trigger_analysis(self):
        """Trigger manual analysis"""
        events = list(self.tracker.events)
        if events or self.conversation_history:
            analysis_request = {
                'id': str(uuid.uuid4()),
                'context': self._prepare_analysis_context(events)
            }
            self._schedule_async_task(
                self.async_handler.analysis_queue.put(analysis_request)
            )

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
                self.conversation_text.insert(tk.END, f"\n[{timestamp}] Claude: {update['data']}")
                self.conversation_text.see(tk.END)
                self._update_analysis_text(update['data'])
            elif update['type'] == 'error':
                self._show_error(update['data'])
        except Exception as e:
            self.logger.error(f"Error processing update: {e}")

    def run(self):
        """Start the application"""
        try:
            self.root.mainloop()
        finally:
            # Clean up async resources
            self.stop_recording()
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            self.async_thread.join(timeout=1.0)
            if self.loop.is_running():
                self.loop.close()

def main():
    logger = setup_logging()
    try:
        API_KEY = "API_KEY_HERE"
        app = WorkflowAnalyzerUI(API_KEY)
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
    finally:
        logging.shutdown()

if __name__ == "__main__":
    main()