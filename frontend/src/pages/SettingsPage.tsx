import SettingsForm from "../components/SettingsForm";

interface SettingsPageProps {
  theme: "light" | "dark";
  onToggleTheme: () => void;
  onClearHistory: () => void;
  addToast: (message: string, type: "success" | "error" | "info") => void;
}

export default function SettingsPage({
  theme,
  onToggleTheme,
  onClearHistory,
  addToast,
}: SettingsPageProps) {
  const handleClearHistory = () => {
    onClearHistory();
    addToast("History cleared", "info");
  };

  return (
    <div className="max-w-2xl">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
          Settings
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Configure your application preferences.
        </p>
      </div>

      <SettingsForm
        theme={theme}
        onToggleTheme={onToggleTheme}
        onClearHistory={handleClearHistory}
      />
    </div>
  );
}
