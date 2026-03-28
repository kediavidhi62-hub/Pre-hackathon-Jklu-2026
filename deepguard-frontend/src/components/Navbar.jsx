import { useState, useEffect } from 'react'
import { Shield } from 'lucide-react'

export default function Navbar() {
    const [scrolled, setScrolled] = useState(false)

    useEffect(() => {
        const onScroll = () => setScrolled(window.scrollY > 20)
        window.addEventListener('scroll', onScroll)
        return () => window.removeEventListener('scroll', onScroll)
    }, [])

    return (
        <nav
            className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? 'glass shadow-lg shadow-black/5' : 'bg-white/0'
                }`}
        >
            <div className="max-w-7xl mx-auto px-6 py-4 flex items-center">
                {/* Logo */}
                <a href="#" className="flex items-center gap-2.5 group">
                    <div className="w-9 h-9 rounded-xl gradient-btn flex items-center justify-center shadow-md shadow-brand-blue/30 group-hover:shadow-lg transition-shadow">
                        <Shield className="w-5 h-5 text-white" />
                    </div>
                    <span className="text-xl font-bold text-black">
                        Deep<span className="gradient-text">Guard</span>
                    </span>
                </a>
            </div>
        </nav>
    )
}
