import { SunIcon, MoonIcon } from "@heroicons/react/24/outline";

interface HeaderProps {
  theme: "light" | "dark";
  onToggleTheme: () => void;
}

export default function Header({ theme, onToggleTheme }: HeaderProps) {
  return (
    <header className="h-16 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between px-6 lg:px-8">
      <div>
        <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
          Intelligent Support Ticket Classification
        </h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 hidden sm:block">
          RAG-powered customer support automation
        </p>
      </div>
      <button
        onClick={onToggleTheme}
        className="p-2 rounded-lg text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        aria-label="Toggle theme"
      >
        {theme === "dark" ? (
          <SunIcon className="w-5 h-5" />
        ) : (
          <MoonIcon className="w-5 h-5" />
        )}
      </button>
    </header>
  );
}
