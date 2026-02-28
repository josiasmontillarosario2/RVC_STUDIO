"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { forwardRef } from "react";
import type { AnchorHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

interface NavLinkCompatProps extends Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "href" | "className"> {
  to: string;
  className?: string;
  activeClassName?: string;
  pendingClassName?: string;
  exact?: boolean;
}

const NavLink = forwardRef<HTMLAnchorElement, NavLinkCompatProps>(
  ({ className, activeClassName, pendingClassName: _pendingClassName, to, exact = false, ...props }, ref) => {
    const pathname = usePathname();
    const isActive = exact ? pathname === to : pathname === to || pathname?.startsWith(`${to}/`);

    return (
      <Link
        ref={ref}
        href={to}
        className={cn(className, isActive && activeClassName)}
        {...props}
      />
    );
  },
);

NavLink.displayName = "NavLink";

export { NavLink };
