import { NavLink } from "react-router-dom";
import {
  HomeModernIcon,
  SparklesIcon,
  ClockIcon,
  Cog6ToothIcon,
} from "@heroicons/react/24/outline";

const navItems = [
  { to: "/", label: "Home", icon: HomeModernIcon },
  { to: "/classify", label: "Classify", icon: SparklesIcon },
  { to: "/history", label: "History", icon: ClockIcon },
  { to: "/settings", label: "Settings", icon: Cog6ToothIcon },
];

export default function Sidebar() {
  return (
    <aside className="fixed inset-y-0 left-0 z-30 w-64 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 flex flex-col">
      <div className="flex items-center gap-3 px-6 h-16 border-b border-gray-200 dark:border-gray-800">
        <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center">
          <SparklesIcon className="w-5 h-5 text-white" />
        </div>
        <span className="font-semibold text-gray-900 dark:text-white text-sm">
          Ticket RAG
        </span>
      </div>
      <nav className="flex-1 px-3 py-4 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? "bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"
                  : "text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
              }`
            }
          >
            <item.icon className="w-5 h-5" />
            {item.label}
          </NavLink>
        ))}
      </nav>
      <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-800">
        <p className="text-xs text-gray-400">v1.0.0</p>
      </div>
    </aside>
  );
}
