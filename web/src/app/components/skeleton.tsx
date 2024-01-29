import { cn } from "@/app/utils/cn";
import { HTMLAttributes } from "react";

function Skeleton({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("animate-pulse rounded-md bg-muted h-32", className)}
      {...props}
    />
  );
}

export { Skeleton };
