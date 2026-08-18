import { useState, useEffect } from 'react'
import { curriculumData } from './data'
import './App.css'

function App() {
  const [pipeline, setPipeline] = useState('cpp') // 'cpp' or 'python'
  const [activeTopic, setActiveTopic] = useState(null)
  const [visibleNodes, setVisibleNodes] = useState(new Set())

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setVisibleNodes((prev) => {
              const newSet = new Set(prev)
              newSet.add(entry.target.dataset.id)
              return newSet
            })
            // Stop observing once it's visible so it doesn't fade out if we scroll back up
            observer.unobserve(entry.target)
          }
        })
      },
      { threshold: 0.1 }
    )

    document.querySelectorAll('.topic-node').forEach((node) => {
      observer.observe(node)
    })

    return () => observer.disconnect()
  }, [])

  return (
    <div className="app-container">
      <header className="hero">
        <h1 className="glow-text">LLM Training Engine</h1>
        <p className="subtitle">The Journey from Raw Text to a Reasoning Matrix</p>
        
        <div className="pipeline-toggle">
          <button 
            type="button"
            className={`toggle-btn ${pipeline === 'cpp' ? 'active' : ''}`}
            onClick={() => setPipeline('cpp')}
          >
            Native C++ (Metal)
          </button>
          <button 
            type="button"
            className={`toggle-btn ${pipeline === 'python' ? 'active' : ''}`}
            onClick={() => setPipeline('python')}
          >
            Python (MLX)
          </button>
        </div>
      </header>

      <main className="curriculum-pathway">
        {curriculumData.map((phase) => (
          <section key={phase.phase} className="phase-section">
            <div className="phase-header">
              <h2>{phase.phase}</h2>
              <p>{phase.description}</p>
            </div>
            
            <div className="topics-container">
              {phase.topics.map((topic) => (
                <div 
                  key={topic.id} 
                  data-id={topic.id}
                  className={`topic-node ${activeTopic === topic.id ? 'expanded' : ''} ${visibleNodes.has(topic.id) ? 'visible' : ''}`}
                >
                  <button 
                    type="button"
                    className="node-marker trigger-button"
                    onClick={() => setActiveTopic(activeTopic === topic.id ? null : topic.id)}
                    aria-label={`Expand ${topic.title}`}
                  >
                    <div className="core"></div>
                    <div className="pulse"></div>
                  </button>
                  
                  <div className="node-content">
                    <button 
                      type="button"
                      className="node-header-trigger trigger-button"
                      onClick={() => setActiveTopic(activeTopic === topic.id ? null : topic.id)}
                      aria-expanded={activeTopic === topic.id}
                    >
                      <div className="node-header">
                        <span className="topic-id">{topic.id}</span>
                        <h3>{topic.title}</h3>
                      </div>
                      <p className="topic-subtitle">{topic.subtitle}</p>
                    </button>
                    
                    {activeTopic === topic.id && (
                      <div className="node-details animate-in">
                        <div className="detail-block">
                          <h4>The Reality</h4>
                          <p>{topic.description}</p>
                        </div>
                        <div className="detail-block highlight">
                          <h4>The "Why"</h4>
                          <p>{topic.why}</p>
                        </div>
                        
                        <a 
                          href={`https://github.com/rahul238xaviers/large-language-model/tree/main/${pipeline === 'cpp' ? topic.cppLink : topic.pythonLink}`}
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="code-link-btn"
                        >
                          View {pipeline === 'cpp' ? 'C++' : 'Python'} Implementation →
                        </a>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </section>
        ))}
      </main>
      
      <footer>
        <p>Built with curiosity. Explore the <a href="https://github.com/rahul238xaviers/large-language-model" target="_blank" rel="noopener noreferrer">GitHub Repository</a>.</p>
      </footer>
    </div>
  )
}

export default App
