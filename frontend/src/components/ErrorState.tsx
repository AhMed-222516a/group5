import { ExclamationTriangleIcon } from "@heroicons/react/24/outline";

interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
}

export default function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className="w-14 h-14 rounded-full bg-red-50 dark:bg-red-900/20 flex items-center justify-center mb-4">
        <ExclamationTriangleIcon className="w-7 h-7 text-red-500" />
      </div>
      <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
        Something went wrong
      </h3>
      <p className="mt-1 text-sm text-gray-500 dark:text-gray-400 max-w-md">
        {message}
      </p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-4 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors"
        >
          Try Again
        </button>
      )}
    </div>
  );
}
