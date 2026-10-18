import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Bridge Lift Explorer',
  description: 'Inspect strict 0/8 to 8/8 bridge interventions and their rollout evidence.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
