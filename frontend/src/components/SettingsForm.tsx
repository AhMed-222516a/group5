import { useState } from "react";
import { SunIcon, MoonIcon, LinkIcon, TrashIcon } from "@heroicons/react/24/outline";
import { API_BASE_KEY, DEFAULT_API_BASE } from "../services/api";

interface SettingsFormProps {
  theme: "light" | "dark";
  onToggleTheme: () => void;
  onClearHistory: () => void;
}

export default function SettingsForm({
  theme,
  onToggleTheme,
  onClearHistory,
}: SettingsFormProps) {
  const [apiUrl, setApiUrl] = useState(
    () => localStorage.getItem(API_BASE_KEY) || DEFAULT_API_BASE
  );
  const [saved, setSaved] = useState(false);

  const handleSaveUrl = () => {
    localStorage.setItem(API_BASE_KEY, apiUrl);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="max-w-2xl space-y-6">
      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm p-6">
        <h3 className="text-base font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Appearance
        </h3>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {theme === "dark" ? (
              <MoonIcon className="w-5 h-5 text-gray-400" />
            ) : (
              <SunIcon className="w-5 h-5 text-gray-500" />
            )}
            <div>
              <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                {theme === "dark" ? "Dark Mode" : "Light Mode"}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Toggle the color theme
              </p>
            </div>
          </div>
          <button
            onClick={onToggleTheme}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
              theme === "dark" ? "bg-blue-600" : "bg-gray-300"
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                theme === "dark" ? "translate-x-6" : "translate-x-1"
              }`}
            />
          </button>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm p-6">
        <h3 className="text-base font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Backend API URL
        </h3>
        <div className="flex items-center gap-3">
          <div className="relative flex-1">
            <LinkIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="url"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="http://localhost:8000"
              className="block w-full pl-10 pr-4 py-2.5 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-colors"
            />
          </div>
          <button
            onClick={handleSaveUrl}
            className="px-4 py-2.5 rounded-lg text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 transition-colors"
          >
            {saved ? "Saved!" : "Save"}
          </button>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm p-6">
        <h3 className="text-base font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Data Management
        </h3>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
              Clear History
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Remove all saved predictions
            </p>
          </div>
          <button
            onClick={onClearHistory}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
          >
            <TrashIcon className="w-4 h-4" />
            Clear
          </button>
        </div>
      </div>
    </div>
  );
}
