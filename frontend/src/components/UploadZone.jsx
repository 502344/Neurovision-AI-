import { useRef, useState } from 'react'

const OK = ['image/jpeg', 'image/png', 'image/tiff']
const MAX_MB = 15

export default function UploadZone({ onFile, onReject, disabled }) {
  const input = useRef(null)
  const [over, setOver] = useState(false)

  const take = (file) => {
    if (!file) return
    if (!OK.includes(file.type)) return onReject(`Unsupported file type "${file.type || 'unknown'}". Use JPG, PNG or TIFF.`)
    if (file.size > MAX_MB * 1024 * 1024) return onReject(`File is too large. The limit is ${MAX_MB} MB.`)
    onFile(file)
  }

  return (
    <div
      className={`dropzone ${over ? 'over' : ''} ${disabled ? 'disabled' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setOver(true) }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); !disabled && take(e.dataTransfer.files?.[0]) }}
      onClick={() => !disabled && input.current?.click()}
      role="button" tabIndex={0}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && !disabled && input.current?.click()}
    >
      <input ref={input} type="file" accept=".jpg,.jpeg,.png,.tif,.tiff,image/jpeg,image/png,image/tiff" hidden
             onChange={(e) => { take(e.target.files?.[0]); e.target.value = '' }} />
      <p className="dz-title">Drop a retinal fundus image here</p>
      <p className="dz-sub">or click to browse · JPG, PNG, TIFF · up to {MAX_MB} MB</p>
    </div>
  )
}
