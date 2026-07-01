import { XMarkIcon, CheckCircleIcon, ExclamationCircleIcon, InformationCircleIcon } from "@heroicons/react/24/outline";
import type { Toast } from "../hooks/useToast";

interface ToastContainerProps {
  toasts?: Toast[];
  onRemove?: (id: string) => void;
}

export default function ToastContainer({
  toasts = [],
  onRemove,
}: ToastContainerProps) {
  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-3 max-w-sm">
      {toasts.map((toast) => {
        const icons = {
          success: CheckCircleIcon,
          error: ExclamationCircleIcon,
          info: InformationCircleIcon,
        };
        const colors = {
          success: "bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-800 text-green-800 dark:text-green-200",
          error: "bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-800 text-red-800 dark:text-red-200",
          info: "bg-blue-50 dark:bg-blue-900/30 border-blue-200 dark:border-blue-800 text-blue-800 dark:text-blue-200",
        };
        const Icon = icons[toast.type];
        return (
          <div
            key={toast.id}
            className={`flex items-start gap-3 px-4 py-3 rounded-lg border shadow-lg text-sm animate-slide-up ${colors[toast.type]}`}
          >
            <Icon className="w-5 h-5 shrink-0 mt-0.5" />
            <span className="flex-1">{toast.message}</span>
            {onRemove && (
              <button onClick={() => onRemove(toast.id)} className="shrink-0 p-0.5 rounded hover:bg-black/5 dark:hover:bg-white/5">
                <XMarkIcon className="w-4 h-4" />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
