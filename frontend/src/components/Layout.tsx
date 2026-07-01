import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import Header from "./Header";
import ToastContainer from "./ToastContainer";

interface LayoutProps {
  theme: "light" | "dark";
  onToggleTheme: () => void;
}

export default function Layout({ theme, onToggleTheme }: LayoutProps) {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100">
      <Sidebar />
      <div className="lg:pl-64">
        <Header theme={theme} onToggleTheme={onToggleTheme} />
        <main className="p-6 lg:p-8">
          <Outlet />
        </main>
      </div>
      <ToastContainer />
    </div>
  );
}
