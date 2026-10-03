import * as React from "react";
import { Eye, EyeOff, Lock, MessageSquare, User } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { register as registerRequest } from "@/services/authService";
import { useAuthStore } from "@/stores/authStore";

type AuthMode = "login" | "register";

interface UserAuthPageProps {
  initialMode: AuthMode;
}

export function UserAuthPage({ initialMode }: UserAuthPageProps) {
  const navigate = useNavigate();
  const { login, isLoading } = useAuthStore();
  const [mode, setMode] = React.useState<AuthMode>(initialMode);
  const [showPassword, setShowPassword] = React.useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = React.useState(false);
  const [form, setForm] = React.useState({
    username: "",
    password: "",
    confirmPassword: ""
  });
  const [error, setError] = React.useState<string | null>(null);
  const [isRegistering, setIsRegistering] = React.useState(false);

  React.useEffect(() => {
    setMode(initialMode);
    setError(null);
  }, [initialMode]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);

    const username = form.username.trim();
    const password = form.password.trim();
    const confirmPassword = form.confirmPassword.trim();

    if (!username || !password) {
      setError("请输入用户名和密码。");
      return;
    }
    if (mode === "register" && password !== confirmPassword) {
      setError("两次输入的密码不一致。");
      return;
    }

    try {
      if (mode === "register") {
        setIsRegistering(true);
        await registerRequest(username, password);
        await login(username, password);
      } else {
        await login(username, password);
      }
      const role = useAuthStore.getState().user?.role;
      navigate(role === "admin" ? "/admin/dashboard" : "/chat", { replace: true });
    } catch (err) {
      setError((err as Error).message || (mode === "register" ? "注册失败，请稍后重试。" : "登录失败，请稍后重试。"));
    } finally {
      setIsRegistering(false);
    }
  };

  const submitting = isLoading || isRegistering;
  const isRegister = mode === "register";

  return (
    <div className="flex min-h-screen items-center justify-center bg-[#F7F8FA] px-4 py-8">
      <div className="w-full max-w-[420px] rounded-lg border border-[#E5E7EB] bg-white p-6 shadow-[0_12px_30px_rgba(15,23,42,0.08)] sm:p-8">
        <div className="mb-6 flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-[#2563EB] text-white shadow-[0_8px_18px_rgba(37,99,235,0.22)]">
            <MessageSquare className="h-5 w-5" />
          </div>
          <div>
            <p className="text-xl font-semibold text-[#111827]">
              {isRegister ? "用户注册" : "用户登录"}
            </p>
            <p className="mt-0.5 text-sm text-[#6B7280]">登录后进入聊天窗口</p>
          </div>
        </div>

        <div className="mb-5 grid grid-cols-2 rounded-lg bg-[#F3F4F6] p-1">
          <button
            type="button"
            onClick={() => {
              setMode("login");
              setError(null);
              navigate("/user/login", { replace: true });
            }}
            className={`rounded-md px-3 py-2 text-sm font-medium transition-colors ${
              !isRegister ? "bg-white text-[#111827] shadow-sm" : "text-[#6B7280] hover:text-[#111827]"
            }`}
          >
            登录
          </button>
          <button
            type="button"
            onClick={() => {
              setMode("register");
              setError(null);
              navigate("/user/register", { replace: true });
            }}
            className={`rounded-md px-3 py-2 text-sm font-medium transition-colors ${
              isRegister ? "bg-white text-[#111827] shadow-sm" : "text-[#6B7280] hover:text-[#111827]"
            }`}
          >
            注册
          </button>
        </div>

        <form className="space-y-4" onSubmit={handleSubmit}>
          <div className="space-y-2">
            <label className="text-sm font-medium text-[#374151]">用户名</label>
            <div className="relative">
              <User className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#9CA3AF]" />
              <Input
                placeholder="请输入用户名"
                value={form.username}
                onChange={(event) => setForm((prev) => ({ ...prev, username: event.target.value }))}
                className="h-11 rounded-lg border-[#D1D5DB] bg-white pl-10 shadow-none focus-visible:ring-[#93C5FD]"
                autoComplete="username"
              />
            </div>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-[#374151]">密码</label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#9CA3AF]" />
              <Input
                type={showPassword ? "text" : "password"}
                placeholder="请输入密码"
                value={form.password}
                onChange={(event) => setForm((prev) => ({ ...prev, password: event.target.value }))}
                className="h-11 rounded-lg border-[#D1D5DB] bg-white pl-10 pr-10 shadow-none focus-visible:ring-[#93C5FD]"
                autoComplete={isRegister ? "new-password" : "current-password"}
              />
              <button
                type="button"
                onClick={() => setShowPassword((prev) => !prev)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-[#9CA3AF] transition-colors hover:text-[#4B5563]"
                aria-label="显示或隐藏密码"
              >
                {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>
          {isRegister ? (
            <div className="space-y-2">
              <label className="text-sm font-medium text-[#374151]">确认密码</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#9CA3AF]" />
                <Input
                  type={showConfirmPassword ? "text" : "password"}
                  placeholder="请再次输入密码"
                  value={form.confirmPassword}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, confirmPassword: event.target.value }))
                  }
                  className="h-11 rounded-lg border-[#D1D5DB] bg-white pl-10 pr-10 shadow-none focus-visible:ring-[#93C5FD]"
                  autoComplete="new-password"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword((prev) => !prev)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-[#9CA3AF] transition-colors hover:text-[#4B5563]"
                  aria-label="显示或隐藏确认密码"
                >
                  {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>
          ) : null}
          {error ? <p className="text-sm text-[#DC2626]">{error}</p> : null}
          <Button
            type="submit"
            className="h-11 w-full rounded-lg bg-[#2563EB] shadow-none hover:bg-[#1D4ED8]"
            disabled={submitting}
          >
            {submitting ? (isRegister ? "正在注册..." : "正在登录...") : isRegister ? "注册并登录" : "登录"}
          </Button>
        </form>

        <div className="mt-5 text-center text-xs text-[#6B7280]">
          <Link to="/login" className="font-medium text-[#2563EB] hover:underline">
            管理员登录
          </Link>
        </div>
      </div>
    </div>
  );
}
