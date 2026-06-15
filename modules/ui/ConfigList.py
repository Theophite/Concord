import contextlib
import copy
import json
import os
import tkinter as tk
from abc import ABCMeta, abstractmethod

from modules.util import path_util
from modules.util.config.BaseConfig import BaseConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.path_util import write_json_atomic
from modules.util.ui import components, dialogs
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class ConfigList(metaclass=ABCMeta):

    def __init__(
            self,
            master,
            train_config: TrainConfig,
            ui_state: UIState,
            from_external_file: bool,
            attr_name: str = "",
            enable_key: str = "enabled",
            config_dir: str = "",
            default_config_name: str = "",
            add_button_text: str = "",
            add_button_tooltip: str = "",
            is_full_width: bool = "",
            show_toggle_button: bool = False,
            page_size: int = None,
    ):
        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state
        self.from_external_file = from_external_file
        self.attr_name = attr_name
        self.enable_key = enable_key

        self.config_dir = config_dir
        self.default_config_name = default_config_name

        self.is_full_width = is_full_width

        # From search-concepts
        self.filters = {"search": "", "type": "ALL", "show_disabled": True}
        self.widgets_initialized = False

        # From master
        self.toggle_button = None
        self.show_toggle_button = show_toggle_button
        self.is_opening_window = False
        self._is_current_item_enabled = False

        # Pagination: page_size set -> render the list in pages of page_size rows
        # (off-page widgets stay None, so widgets[i] still maps to current_config[i],
        # and only ~page_size rows + their Tk menus exist at once). Default None =
        # render every row (original behavior); only AdditionalEmbeddingsTab opts in.
        self.page_size = page_size
        self.page = 0
        self.pagination_frame = None
        self.page_label = None
        self.prev_button = None
        self.next_button = None

        self.master.grid_rowconfigure(0, weight=0)
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        if self.from_external_file:
            self.top_frame = ctk.CTkFrame(self.master, fg_color="transparent")
            self.top_frame.grid(row=0, column=0, sticky="nsew")

            self.configs_dropdown = None
            self.element_list = None

            self.configs = []
            self.__load_available_config_names()

            self.current_config = getattr(self.train_config, self.attr_name)
            self.widgets = []
            self.__load_current_config(getattr(self.train_config, self.attr_name))

            self.__create_configs_dropdown()
            components.button(self.top_frame, 0, 1, "Add Config", self.__add_config, tooltip="Adds a new config, which are containers for concepts, which themselves contain your dataset", width=20, padx=5)
            components.button(self.top_frame, 0, 2, add_button_text, self.__add_element, tooltip=add_button_tooltip, width=30, padx=5)
        else:
            self.top_frame = ctk.CTkFrame(self.master, fg_color="transparent")
            self.top_frame.grid(row=0, column=0, sticky="nsew")
            components.button(self.top_frame, 0, 2, add_button_text, self.__add_element, width=20, padx=5)

            self.current_config = getattr(self.train_config, self.attr_name)

            self.element_list = None
            self._create_element_list()

        if show_toggle_button:
            # tooltips break if you initialize with an empty string, default to a single space
            self.toggle_button = components.button(self.top_frame, 0, 3, " ", self._toggle, tooltip="Disables/Enables all visible items in the current view", width=30, padx=5)
            self._update_toggle_button_text()

        if self.page_size is not None and not self.from_external_file:
            self._create_pagination_controls()



    @abstractmethod
    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        pass

    @abstractmethod
    def create_new_element(self) -> BaseConfig:
        pass

    @abstractmethod
    def open_element_window(self, i, ui_state) -> ctk.CTkToplevel:
        pass

    def _refresh_show_disabled_text(self):
        return

    def _reset_filters(self):  # pragma: no cover - default noop
        search_var = getattr(self, 'search_var', None)
        filter_var = getattr(self, 'filter_var', None)
        show_disabled_var = getattr(self, 'show_disabled_var', None)

        if search_var:
            search_var.set("")
        if filter_var:
            filter_var.set("ALL")
        if show_disabled_var:
            show_disabled_var.set(True)
        if search_var and hasattr(self, '_update_filters'):
            self._update_filters()

    def _update_item_enabled_state(self):
        # Only count items that match current filters
        self._is_current_item_enabled = any(
            item.ui_state.get_var(self.enable_key).get()
            for i, item in enumerate(self.widgets)
            if item is not None and i < len(self.current_config) and self._element_matches_filters(self.current_config[i])
        )

    def _update_toggle_button_text(self):
        if not self.show_toggle_button:
            return
        self._update_item_enabled_state()
        if self.toggle_button is not None:
            self.toggle_button.configure(text="Disable" if self._is_current_item_enabled else "Enable")

    def _toggle(self):
        self._toggle_items()

    def _toggle_items(self):
        enable_state = not self._is_current_item_enabled

        # Only toggle items that match current filters (None = off-page, skip)
        for i, widget in enumerate(self.widgets):
            if widget is not None and i < len(self.current_config) and self._element_matches_filters(self.current_config[i]):
                widget.ui_state.get_var(self.enable_key).set(enable_state)
        self.save_current_config()

        self._update_widget_visibility()

    def __create_configs_dropdown(self):
        if self.configs_dropdown is not None:
            self.configs_dropdown.destroy()

        self.configs_dropdown = components.options_kv(
            self.top_frame, 0, 0, self.configs, self.ui_state, self.attr_name, self.__load_current_config
        )
        self._update_toggle_button_text()

    def _create_element_list(self, **filters):
        if not self.from_external_file:
            self.current_config = getattr(self.train_config, self.attr_name)

        self.filters.update(filters)

        if not self.widgets_initialized:
            self._initialize_all_widgets()
            self.widgets_initialized = True

        self._update_widget_visibility()
        self._update_toggle_button_text()
        if self.page_size is not None:
            self._update_pagination_controls()

    def _initialize_all_widgets(self):
        self.widgets = []
        if self.element_list is not None:
            self.element_list.destroy()

        self.element_list = ctk.CTkScrollableFrame(self.master, fg_color="transparent")
        self.element_list.grid(row=1, column=0, sticky="nsew")

        if self.is_full_width:
            self.element_list.grid_columnconfigure(0, weight=1)

        if self.page_size is not None:
            self._build_page_widgets()
        else:
            for i, element in enumerate(self.current_config):
                widget = self.create_widget(
                    self.element_list, element, i,
                    self.__open_element_window,
                    self.__remove_element,
                    self.__clone_element,
                    self.save_current_config
                )
                self.widgets.append(widget)

    def _page_count(self):
        return max(1, (len(self.current_config) + self.page_size - 1) // self.page_size)

    def _build_page_widgets(self):
        """Create widgets ONLY for the current page's slice of current_config; the
        rest of self.widgets stays None so index i still maps to current_config[i]
        and only ~page_size rows (and their Tk menus) exist at once."""
        total = len(self.current_config)
        self.page = max(0, min(self.page, self._page_count() - 1))
        start = self.page * self.page_size
        end = min(start + self.page_size, total)
        self.widgets = [None] * total
        for i in range(start, end):
            self.widgets[i] = self.create_widget(
                self.element_list, self.current_config[i], i,
                self.__open_element_window,
                self.__remove_element,
                self.__clone_element,
                self.save_current_config,
            )

    def _render_page(self):
        """Re-render the current page in place: free the old rows (and their Tk
        menus), rebuild the page slice, refresh visibility + the page label."""
        for w in self.widgets:
            if w is not None:
                with contextlib.suppress(tk.TclError, AttributeError):
                    w.destroy()
        self._build_page_widgets()
        self._update_widget_visibility()
        self._update_pagination_controls()
        self._update_toggle_button_text()

    def _create_pagination_controls(self):
        self.pagination_frame = ctk.CTkFrame(self.master, fg_color="transparent")
        self.pagination_frame.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        self.prev_button = components.button(
            self.pagination_frame, 0, 0, "< prev", lambda: self._change_page(-1), width=20, padx=5)
        self.page_label = ctk.CTkLabel(self.pagination_frame, text="")
        self.page_label.grid(row=0, column=1, padx=10)
        self.next_button = components.button(
            self.pagination_frame, 0, 2, "next >", lambda: self._change_page(1), width=20, padx=5)
        self._update_pagination_controls()

    def _change_page(self, delta: int):
        new_page = max(0, min(self.page + delta, self._page_count() - 1))
        if new_page != self.page:
            self.page = new_page
            self._render_page()

    def _update_pagination_controls(self):
        if self.page_label is None:
            return
        total = len(self.current_config)
        start = self.page * self.page_size + 1 if total else 0
        end = min((self.page + 1) * self.page_size, total)
        self.page_label.configure(
            text=f"page {self.page + 1} / {self._page_count()}    ({start}-{end} of {total})")

    def _update_widget_visibility(self):
        visible_index = 0

        for i, widget in enumerate(self.widgets):
            if widget is not None and i < len(self.current_config):
                element = self.current_config[i]

                if self._element_matches_filters(element):
                    widget.visible_index = visible_index
                    widget.place_in_list()
                    visible_index += 1
                else:
                    widget.grid_remove()

    def __load_available_config_names(self):
        if os.path.isdir(self.config_dir):
            for path in os.listdir(self.config_dir):
                path = path_util.canonical_join(self.config_dir, path)
                if path.endswith(".json") and os.path.isfile(path):
                    name = os.path.basename(path)
                    name = os.path.splitext(name)[0]
                    self.configs.append((name, path))

        # Keep an externally-referenced file (e.g. an absolute concept_file_name carried in by a
        # loaded preset that points outside config_dir) selectable, so the dropdown can hold it and
        # options_kv() does not reset it to '' / the first entry. Without this the tab desyncs from
        # the concepts/samples the loaded config actually points at.
        current = getattr(self.train_config, self.attr_name, "") or ""
        if current and os.path.isfile(current) and not any(
                os.path.normcase(os.path.abspath(p)) == os.path.normcase(os.path.abspath(current))
                for _, p in self.configs):
            self.configs.append((os.path.splitext(os.path.basename(current))[0], current))

        if len(self.configs) == 0:
            name = self.default_config_name.removesuffix(".json")
            self.__create_config(name)
            self.save_current_config()

    def __create_config(self, name: str):
        name = path_util.safe_filename(name)
        path = path_util.canonical_join(self.config_dir, f"{name}.json")
        self.configs.append((name, path))
        self.__create_configs_dropdown()

    def __add_config(self):
        dialogs.StringInputDialog(self.master, "name", "Name", self.__create_config)

    def __add_element(self):
        new_element = self.create_new_element()
        self.current_config.append(new_element)
        # paginated: jump to the page holding the new row and re-render it
        if self.page_size is not None:
            self.page = (len(self.current_config) - 1) // self.page_size
            self._render_page()
            self.save_current_config()
            return
        # incremental insertion if widgets already initialized, else fall back to full rebuild
        if self.widgets_initialized and self.element_list is not None:
            i = len(self.current_config) - 1
            widget = self.create_widget(
                self.element_list, new_element, i,
                self.__open_element_window,
                self.__remove_element,
                self.__clone_element,
                self.save_current_config
            )
            self.widgets.append(widget)
            self._update_widget_visibility()
        else:
            self.widgets_initialized = False
            self._create_element_list()
        self.save_current_config()

    def __clone_element(self, clone_i, modify_element_fun=None):
        new_element = copy.deepcopy(self.current_config[clone_i])

        if modify_element_fun is not None:
            new_element = modify_element_fun(new_element)
        self.current_config.append(new_element)
        # paginated: jump to the page holding the clone and re-render it
        if self.page_size is not None:
            self.page = (len(self.current_config) - 1) // self.page_size
            self._render_page()
            self.save_current_config()
            return
        if self.widgets_initialized and self.element_list is not None:
            i = len(self.current_config) - 1
            widget = self.create_widget(
                self.element_list, new_element, i,
                self.__open_element_window,
                self.__remove_element,
                self.__clone_element,
                self.save_current_config
            )
            self.widgets.append(widget)
            self._update_widget_visibility()
        else:
            self.widgets_initialized = False
            self._create_element_list()
        self.save_current_config()

    def __remove_element(self, remove_i):
        self.current_config.pop(remove_i)
        # paginated: re-render the current page (indices shift; page is re-clamped)
        if self.page_size is not None:
            self._render_page()
            self.save_current_config()
            return
        if self.widgets_initialized and 0 <= remove_i < len(self.widgets):
            removed = self.widgets.pop(remove_i)
            with contextlib.suppress(tk.TclError, AttributeError):
                removed.destroy()
            # Reindex remaining widgets
            for idx, widget in enumerate(self.widgets):
                widget.i = idx
            self._update_widget_visibility()
        else:
            self.widgets_initialized = False
            self._create_element_list()
        self.save_current_config()

    def __load_current_config(self, filename):
        try:
            with open(filename, "r") as f:
                self.current_config = []

                loaded_config_json = json.load(f)
                for element_json in loaded_config_json:
                    element = self.create_new_element().from_dict(element_json)
                    self.current_config.append(element)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Failed to load config from {filename}: {e}")
            self.current_config = []

        # reset filters when switching configs
        if hasattr(self, '_reset_filters') and self.widgets_initialized:
            self._reset_filters()

        self.widgets_initialized = False
        self._create_element_list()
        self._update_toggle_button_text()

    def refresh_ui(self):
        """Re-sync this tab with the current train_config (e.g. after a preset load).

        For a file-backed list, re-read the available files (including the currently-referenced
        one even if it lives outside config_dir), reload the element list from the bound file
        path, and rebuild the dropdown so its selection matches train_config.<attr_name>.
        Without this, loading a config leaves the concept/sample tab showing the previous
        entries while the file pointer changes underneath -- the desync that ends in the
        '.write -> ''' save failures. (Re-creating the dropdown is safe: the old widget is
        destroyed and its stale var-trace no-ops via component.winfo_exists().)"""
        if self.from_external_file:
            self.configs = []
            self.__load_available_config_names()
            self.__load_current_config(getattr(self.train_config, self.attr_name))
            self.__create_configs_dropdown()
        else:
            if self.element_list is not None:
                self.element_list.destroy()
                self.element_list = None
            self.widgets_initialized = False
            self._create_element_list()

    def save_current_config(self):
        if self.from_external_file:
            try:
                if not os.path.exists(self.config_dir):
                    os.makedirs(self.config_dir, exist_ok=True)

                write_json_atomic(
                    getattr(self.train_config, self.attr_name),
                    [element.to_dict() for element in self.current_config]
                )
            except (OSError) as e:
                print(f"Failed to save config: {e}")

        self._update_toggle_button_text()

        if self.widgets_initialized:
            try:
                self._update_widget_visibility()
            except (tk.TclError, AttributeError) as e:
                print.debug(f"Widget visibility update failed: {e}")

        # let subclass refresh any show-disabled UI
        if hasattr(self, '_refresh_show_disabled_text'):
            self._refresh_show_disabled_text()

    def _element_matches_filters(self, element):
        return True  # Show all by default

    def __open_element_window(self, i, ui_state):
        if self.is_opening_window:
            return
        self.is_opening_window = True
        try:
            window = self.open_element_window(i, ui_state)
            self.master.wait_window(window)
            try:
                if self.widgets is not None and 0 <= i < len(self.widgets) and self.widgets[i] is not None:
                    self.widgets[i].configure_element()
            except Exception:
                self.widgets_initialized = False
                self._create_element_list()
            self.save_current_config()
        finally:
            self.is_opening_window = False
