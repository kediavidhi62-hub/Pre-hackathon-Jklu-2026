import { useState } from 'react'
import { ChevronDown, Send, MessageCircle } from 'lucide-react'

const FAQ_ITEMS = [
    {
        q: 'What is a deepfake?',
        a: 'A deepfake is AI-generated or AI-manipulated media (images, videos, or audio) that appears genuine but has been synthetically altered. They are created using deep learning techniques like GANs (Generative Adversarial Networks) and can be used for misinformation, fraud, or entertainment.',
    },
    {
        q: 'How are deepfakes created?',
        a: 'Deepfakes are typically created using neural networks, particularly GANs and autoencoders. The AI is trained on thousands of images or audio samples of a target person, learning to generate new content that mimics their appearance, voice, and mannerisms with striking realism.',
    },
    {
        q: 'How can I identify a deepfake manually?',
        a: 'Look for unnatural blinking patterns, inconsistent lighting or shadows, blurry edges around the face, mismatched skin tones, audio-lip sync issues, and unnatural head movements. However, modern deepfakes are increasingly difficult to detect without AI assistance.',
    },
    {
        q: 'Why does deepfake detection matter?',
        a: 'Deepfakes pose serious threats to individuals, organizations, and society — from identity fraud and non-consensual content to political disinformation. Detection tools like DeepGuard help maintain trust in digital media and protect people from harm.',
    },
    {
        q: 'How does DeepGuard work?',
        a: 'DeepGuard uses a multi-layered AI analysis pipeline that examines facial landmarks, pixel-level inconsistencies, GAN artifacts, compression patterns, and metadata anomalies. The results are explainable — we show you exactly why a file was flagged.',
    },
]

export default function FAQ() {
    const [openIndex, setOpenIndex] = useState(null)
    const [question, setQuestion] = useState('')
    const [submitted, setSubmitted] = useState(false)

    const handleSubmit = (e) => {
        e.preventDefault()
        if (question.trim()) {
            setSubmitted(true)
            setTimeout(() => {
                setSubmitted(false)
                setQuestion('')
            }, 3000)
        }
    }

    return (
        <section className="py-20 px-6">
            <div className="max-w-2xl mx-auto">
                <div className="text-center mb-8">
                    <h2 className="text-3xl md:text-4xl font-bold text-navy mb-3">
                        Frequently Asked <span className="gradient-text">Questions</span>
                    </h2>
                    <p className="text-soft-gray">
                        Learn more about deepfakes and how DeepGuard protects you.
                    </p>
                </div>

                {/* Ask a Question Bar */}
                <form onSubmit={handleSubmit} className="mb-8">
                    <div className="flex items-center gap-2 p-2 rounded-xl border border-gray-200 bg-white shadow-sm focus-within:border-brand-blue focus-within:ring-2 focus-within:ring-brand-blue/10 transition-all">
                        <MessageCircle className="w-5 h-5 text-soft-gray ml-2 flex-shrink-0" />
                        <input
                            type="text"
                            placeholder="Ask your question..."
                            value={question}
                            onChange={(e) => setQuestion(e.target.value)}
                            className="flex-1 py-2 text-navy text-sm placeholder-soft-gray/60 focus:outline-none bg-transparent"
                        />
                        <button
                            type="submit"
                            disabled={!question.trim()}
                            className="gradient-btn text-white px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-1.5 disabled:opacity-40 hover:opacity-90 transition-opacity"
                        >
                            <Send className="w-4 h-4" />
                            Ask
                        </button>
                    </div>
                    {submitted && (
                        <p className="text-green-600 text-sm mt-2 text-center animate-fade-in">
                            ✅ Your question has been submitted! We'll get back to you soon.
                        </p>
                    )}
                </form>

                <div className="space-y-3">
                    {FAQ_ITEMS.map((item, i) => (
                        <div key={i} className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
                            <button
                                onClick={() => setOpenIndex(openIndex === i ? null : i)}
                                className="w-full flex items-center justify-between px-6 py-4 text-left"
                            >
                                <span className="font-semibold text-navy text-sm pr-4">{item.q}</span>
                                <ChevronDown
                                    className={`w-5 h-5 text-soft-gray flex-shrink-0 transition-transform duration-300 ${openIndex === i ? 'rotate-180' : ''
                                        }`}
                                />
                            </button>
                            <div
                                className={`overflow-hidden transition-all duration-300 ${openIndex === i ? 'max-h-60 opacity-100' : 'max-h-0 opacity-0'
                                    }`}
                            >
                                <p className="px-6 pb-4 text-soft-gray text-sm leading-relaxed">
                                    {item.a}
                                </p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    )
}
