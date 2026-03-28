import { useEffect, useRef, useState } from 'react'
import { Target, Database, Lock, Trash2, Users } from 'lucide-react'

const STATS = [
    { icon: Target, value: 96, suffix: '%+', label: 'Detection Accuracy' },
    { icon: Database, value: 100, suffix: 'K+', label: 'Dataset Trained' },
    { icon: Users, value: 10, suffix: 'K+', label: 'Users Worldwide' },
]

const TRUST_ITEMS = [
    { icon: Lock, title: 'Secure Processing', desc: 'All file analysis happens in encrypted, isolated sessions.' },
    { icon: Trash2, title: 'Auto-Deleted After Scan', desc: 'Your uploaded files are permanently removed after analysis.' },
    { icon: Users, title: 'Used by Professionals', desc: 'Trusted by journalists, fact-checkers, and students globally.' },
]

function AnimatedCounter({ target, suffix }) {
    const [count, setCount] = useState(0)
    const ref = useRef(null)
    const started = useRef(false)

    useEffect(() => {
        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting && !started.current) {
                    started.current = true
                    let current = 0
                    const step = target / 40
                    const interval = setInterval(() => {
                        current += step
                        if (current >= target) {
                            setCount(target)
                            clearInterval(interval)
                        } else {
                            setCount(Math.floor(current))
                        }
                    }, 30)
                }
            },
            { threshold: 0.5 }
        )
        if (ref.current) observer.observe(ref.current)
        return () => observer.disconnect()
    }, [target])

    return (
        <span ref={ref} className="text-4xl md:text-5xl font-extrabold gradient-text">
            {count}{suffix}
        </span>
    )
}

export default function TrustSection() {
    return (
        <section id="trust" className="py-20 px-6 bg-light-blue">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <div className="text-center mb-14">
                    <h2 className="text-3xl md:text-4xl font-bold text-navy mb-3">
                        Why Trust <span className="gradient-text">DeepGuard</span>?
                    </h2>
                    <p className="text-soft-gray max-w-xl mx-auto">
                        Built on cutting-edge AI research with privacy and transparency at its core.
                    </p>
                </div>

                {/* Animated Counters */}
                <div className="grid md:grid-cols-3 gap-6 mb-14">
                    {STATS.map((stat) => {
                        const Icon = stat.icon
                        return (
                            <div key={stat.label} className="bg-white rounded-2xl border border-gray-100 shadow-sm p-8 text-center">
                                <Icon className="w-8 h-8 text-brand-blue mx-auto mb-4" />
                                <AnimatedCounter target={stat.value} suffix={stat.suffix} />
                                <p className="text-soft-gray text-sm mt-2">{stat.label}</p>
                            </div>
                        )
                    })}
                </div>

                {/* Trust Items */}
                <div className="grid md:grid-cols-3 gap-6">
                    {TRUST_ITEMS.map((item) => {
                        const Icon = item.icon
                        return (
                            <div key={item.title} className="flex items-start gap-4 bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
                                <div className="w-10 h-10 rounded-xl bg-blue-50 flex items-center justify-center flex-shrink-0">
                                    <Icon className="w-5 h-5 text-brand-blue" />
                                </div>
                                <div>
                                    <h4 className="font-semibold text-navy mb-1 text-sm">{item.title}</h4>
                                    <p className="text-soft-gray text-xs leading-relaxed">{item.desc}</p>
                                </div>
                            </div>
                        )
                    })}
                </div>
            </div>
        </section>
    )
}
