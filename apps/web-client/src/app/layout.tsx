import type { Metadata, Viewport } from 'next';
import React from 'react';
import { CitationSyncProvider } from '@/components/ast/AstRenderer';
import './globals.css';

export const metadata: Metadata = {
  title: 'CalmFeed — ИИ-агрегатор Telegram и гибридный Glass-box RAG',
  description:
    'Автономный ридер без шума и дофаминовых счетчиков, верифицируемый Glass-box RAG и локальный поиск FTS5',
  manifest: '/manifest.json',
};

export const viewport: Viewport = {
  themeColor: '#FFFFFF',
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ru" data-theme="light">
      <body className="app-shell">
        <CitationSyncProvider>{children}</CitationSyncProvider>
      </body>
    </html>
  );
}
