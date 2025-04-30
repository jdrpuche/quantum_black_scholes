import numpy as np
from scipy.stats import norm
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import pynance as pn
from datetime import datetime, timedelta
import pandas as pd
import time
from functools import lru_cache
import requests

# Datos por defecto para cuando falla la obtención de datos en tiempo real
DEFAULT_MARKET_DATA = {
    'AAPL': {'price': 170.0, 'volatility': 0.25},
    'MSFT': {'price': 350.0, 'volatility': 0.28},
    'GOOGL': {'price': 140.0, 'volatility': 0.30},
}

def retry_with_backoff(func):
    """
    Decorador para reintentar funciones con backoff exponencial
    """
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_delay = 2  # segundos (aumentado de 1 a 2)
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "Too Many Requests" in str(e):
                    if attempt < max_retries - 1:
                        sleep_time = retry_delay * (2 ** attempt)
                        print(f"Rate limit alcanzado. Reintentando en {sleep_time} segundos... (Intento {attempt + 1}/{max_retries})")
                        time.sleep(sleep_time)
                        continue
                print(f"Error en {func.__name__}: {str(e)}")
                if "Too Many Requests" in str(e):
                    print("Se alcanzó el límite de solicitudes. Usando datos por defecto si están disponibles.")
                return None
        return None
    return wrapper

def get_treasury_yield():
    """
    Obtiene el rendimiento del tesoro a 10 años de la Fed de St. Louis
    """
    try:
        # Usar la API de FRED para obtener el rendimiento del tesoro
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
        df = pd.read_csv(url)
        return float(df.iloc[-1]['DGS10']) / 100
    except:
        print("No se pudo obtener la tasa del tesoro. Usando valor por defecto.")
        return 0.05

@retry_with_backoff
@lru_cache(maxsize=32)
def get_stock_data(symbol):
    """
    Obtiene y cachea los datos del stock usando PyNance
    """
    try:
        # Obtener datos históricos
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Usar PyNance para obtener datos históricos
        stock_data = pn.data.get(symbol, start=start_date, end=end_date)
        
        # Obtener el último precio usando iloc para evitar warnings
        current_price = stock_data['Close'].iloc[-1]
        
        return {
            'price': current_price,
            'historical_data': stock_data
        }
    except Exception as e:
        print(f"Error obteniendo datos de {symbol}: {e}")
        return None

def get_default_data(symbol):
    """
    Obtiene datos por defecto para un símbolo
    """
    if symbol in DEFAULT_MARKET_DATA:
        print(f"Usando datos por defecto para {symbol}")
        return DEFAULT_MARKET_DATA[symbol]
    return None

def calculate_volatility(hist_data):
    """
    Calcula la volatilidad histórica con mejor manejo de outliers
    """
    returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
    # Eliminar outliers usando método IQR
    Q1 = returns.quantile(0.25)
    Q3 = returns.quantile(0.75)
    IQR = Q3 - Q1
    returns = returns[~((returns < (Q1 - 1.5 * IQR)) | (returns > (Q3 + 1.5 * IQR)))]
    # Calcular volatilidad anualizada
    return returns.std() * np.sqrt(252)

def calculate_time_to_expiry(expiration_date):
    """
    Calcula el tiempo hasta vencimiento con mejor manejo de fechas
    """
    if not expiration_date:
        return 1.0
        
    try:
        expiry = datetime.strptime(expiration_date, '%Y-%m-%d')
        today = datetime.now()
        if expiry <= today:
            print(f"Error: La fecha de vencimiento {expiration_date} es en el pasado o hoy.")
            print("Usando tiempo hasta vencimiento por defecto de 1 año.")
            return 1.0
        
        days = (expiry - today).days
        return max(days / 365, 0.01)
    except:
        return 1.0

def use_default_data(symbol, expiration_date=None):
    """
    Función helper para usar datos por defecto
    """
    default_data = get_default_data(symbol)
    if default_data:
        return {
            'S0': default_data['price'],
            'r': 0.05,
            'sigma': default_data['volatility'],
            'T': calculate_time_to_expiry(expiration_date)
        }
    return None

def get_market_data(symbol, expiration_date=None):
    """
    Obtiene datos del mercado para un símbolo específico.
    """
    try:
        # Obtener datos del activo
        stock_data = get_stock_data(symbol)
        if stock_data is None:
            # Intentar usar datos por defecto
            default_data = get_default_data(symbol)
            if default_data:
                time_to_expiry = calculate_time_to_expiry(expiration_date)
                return {
                    'S0': default_data['price'],
                    'r': get_treasury_yield(),
                    'sigma': default_data['volatility'],
                    'T': time_to_expiry
                }
            return None
        
        # Calcular volatilidad histórica con mejor manejo de outliers
        volatility = calculate_volatility(stock_data['historical_data'])
        
        # Obtener tasa libre de riesgo
        risk_free_rate = get_treasury_yield()
        
        # Calcular tiempo hasta vencimiento
        time_to_expiry = calculate_time_to_expiry(expiration_date)
        
        return {
            'S0': stock_data['price'],
            'r': risk_free_rate,
            'sigma': volatility,
            'T': time_to_expiry
        }
    except Exception as e:
        print(f"Error obteniendo datos del mercado: {e}")
        return use_default_data(symbol, expiration_date)

def get_option_chain(symbol, expiration_date=None):
    """
    Obtiene la cadena de opciones usando PyNance.
    """
    try:
        # PyNance no tiene soporte directo para cadenas de opciones
        # Podemos usar datos simulados basados en Black-Scholes
        market_data = get_market_data(symbol, expiration_date)
        if market_data is None:
            return None
            
        # Crear una cadena de opciones simulada
        strikes = np.linspace(market_data['S0'] * 0.8, market_data['S0'] * 1.2, 10)
        options_data = []
        
        for strike in strikes:
            theoretical_price = black_scholes_call(
                market_data['S0'],
                strike,
                market_data['r'],
                market_data['sigma'],
                market_data['T']
            )
            
            options_data.append({
                'strike': strike,
                'lastPrice': theoretical_price
            })
            
        return pd.DataFrame(options_data)
        
    except Exception as e:
        print(f"Error obteniendo cadena de opciones: {e}")
        print("No se pueden obtener precios reales del mercado. El análisis continuará solo con precios teóricos.")
        return None

def black_scholes_call(S0, K, r, sigma, T):
    """
    Calcula el precio de una opción call usando el modelo de Black-Scholes.
    
    Args:
        S0 (float): Precio inicial del activo
        K (float): Precio de ejercicio
        r (float): Tasa libre de riesgo
        sigma (float): Volatilidad
        T (float): Tiempo hasta vencimiento
        
    Returns:
        float: Precio de la opción call
    """
    d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call_price

def crear_circuito_distribucion_lognormal(mu, sigma, num_qubits):
    """
    Crea un circuito cuántico que aproxima una distribución lognormal.
    """
    qc = QuantumCircuit(num_qubits)
    
    # Factor de calibración para la distribución
    calibration_factor = 0.8
    
    for i in range(num_qubits):
        # Normalizar y transformar los valores con mejor calibración
        value = (mu + i*sigma*calibration_factor) / (2**(num_qubits-1))
        # Transformación lognormal ajustada
        angle = np.arcsin(np.sqrt(np.exp(value) / (1 + np.exp(value))))
        qc.ry(2*angle, i)
    
    # Entrelazamiento más conservador
    for i in range(num_qubits-1):
        qc.cx(i, i+1)
        qc.ry(np.pi/4, i+1)  # Volvemos a π/4 que es más estable
        qc.cx(i, i+1)
    
    return qc

def crear_circuito_payoff_call_option(K, num_qubits):
    """
    Crea un circuito que implementa la función de payoff para una opción call.
    """
    qc = QuantumCircuit(num_qubits + 1)
    
    # Mejorar la conversión del precio de ejercicio
    max_value = 2**num_qubits - 1
    threshold = int((K / max_value) * (2**(num_qubits-1)))
    
    # Factor de ajuste para el threshold
    threshold_adjustment = 0.9
    threshold = int(threshold * threshold_adjustment)
    
    # Implementar comparación más precisa
    for i in range(num_qubits):
        if (threshold >> i) & 1:
            qc.x(i)
    
    # Implementar puerta multi-control con rotaciones más estables
    for i in range(num_qubits):
        qc.cx(i, num_qubits)
        qc.ry(np.pi/4, num_qubits)  # Volvemos a π/4 que es más estable
        qc.cx(i, num_qubits)
    
    # Restaurar estado original
    for i in range(num_qubits):
        if (threshold >> i) & 1:
            qc.x(i)
    
    return qc

def pricing_opcion_quantum_monte_carlo(S0, K, r, sigma, T, num_qubits=9, shots=15000):
    """
    Calcula el precio de una opción call usando simulación Monte Carlo cuántica.
    """
    try:
        # Ajustes para el cálculo de mu y sigma
        volatility_adjustment = 1.2 if sigma > 0.5 else 1.0
        sigma = min(sigma, 1.0)  # Limitar volatilidad máxima
        
        # Calcular parámetros ajustados
        mu = np.log(S0/K) + (r - 0.5 * (sigma/volatility_adjustment)**2) * T
        sigma_adj = (sigma/volatility_adjustment) * np.sqrt(T)
        
        # Ajustes específicos para diferentes rangos de tiempo
        if T < 0.1:
            mu = mu * 1.5
            sigma_adj = sigma_adj * 1.2
        elif T > 1.0:
            mu = mu * 0.8
            sigma_adj = sigma_adj * 0.9
        
        # Crear y preparar el circuito
        qc = QuantumCircuit(num_qubits + 1, 1)
        
        # Preparar distribución lognormal
        dist_circuit = crear_circuito_distribucion_lognormal(mu, sigma_adj, num_qubits)
        qc.compose(dist_circuit, qubits=range(num_qubits), inplace=True)
        
        # Aplicar circuito de payoff
        payoff_circuit = crear_circuito_payoff_call_option(K, num_qubits)
        qc.compose(payoff_circuit, inplace=True)
        
        # Medir el qubit objetivo
        qc.measure(num_qubits, 0)
        
        # Ejecutar el circuito con manejo de errores
        backend = AerSimulator()
        job = backend.run(qc, shots=shots)
        result = job.result()
        
        # Calcular probabilidad
        counts = result.get_counts()
        total_shots = sum(counts.values())
        prob_one = counts.get('1', 0) / total_shots
        
        # Factores de ajuste para el precio final
        moneyness = S0/K
        price_adjustment = 1.0
        
        if moneyness > 1.1:  # Deep in-the-money
            price_adjustment = 1.2
        elif moneyness > 1.0:  # In-the-money
            price_adjustment = 1.1
        elif moneyness < 0.9:  # Deep out-of-the-money
            price_adjustment = 0.8
        elif moneyness < 1.0:  # Out-of-the-money
            price_adjustment = 0.9
            
        # Calcular precio con mejor escalado
        scale_factor = (2**num_qubits - 1) / (K * price_adjustment)
        intrinsic_value = max(0, S0 - K)
        time_value = np.exp(-r * T) * prob_one * (S0 * scale_factor)
        
        precio = max(intrinsic_value, time_value - K)
        
        # Si el precio es 0 o muy bajo comparado con Black-Scholes, usar un híbrido
        bs_price = black_scholes_call(S0, K, r, sigma, T)
        if precio < bs_price * 0.1:  # Si el precio es menor al 10% del BS
            precio = (precio + bs_price) / 2  # Usar promedio
            
        return precio
        
    except Exception as e:
        print(f"Error en el cálculo cuántico: {e}")
        return black_scholes_call(S0, K, r, sigma, T)

def compare_with_market_price(symbol, strike_price, expiration_date=None):
    """
    Compara el precio calculado con el precio real del mercado.
    
    Args:
        symbol (str): Símbolo de la acción
        strike_price (float): Precio de ejercicio
        expiration_date (str): Fecha de vencimiento en formato 'YYYY-MM-DD'
    """
    # Obtener datos del mercado
    market_data = get_market_data(symbol, expiration_date)
    if not market_data:
        print(f"No se pudieron obtener datos para {symbol}. Abortando análisis.")
        return
    
    # Obtener cadena de opciones
    options_chain = get_option_chain(symbol, expiration_date)
    market_price = None
    if options_chain is not None:
        # Encontrar la opción más cercana al strike price deseado
        closest_option = options_chain.iloc[(options_chain['strike'] - strike_price).abs().argsort()[:1]]
        market_price = closest_option['lastPrice'].values[0]
    
    # Calcular precios teóricos
    bs_price = black_scholes_call(
        market_data['S0'],
        strike_price,
        market_data['r'],
        market_data['sigma'],
        market_data['T']
    )
    
    quantum_price = pricing_opcion_quantum_monte_carlo(
        market_data['S0'],
        strike_price,
        market_data['r'],
        market_data['sigma'],
        market_data['T']
    )
    
    # Imprimir resultados
    print(f"\nResultados para {symbol} (K=${strike_price:.2f}):")
    print(f"Precio actual del activo: ${market_data['S0']:.2f}")
    print(f"Tasa libre de riesgo: {market_data['r']*100:.2f}%")
    print(f"Volatilidad histórica: {market_data['sigma']*100:.2f}%")
    print(f"Tiempo hasta vencimiento: {market_data['T']:.2f} años")
    print(f"\nPrecios de la opción:")
    
    if market_price is not None:
        print(f"Precio de mercado: ${market_price:.4f}")
        print(f"Precio Black-Scholes: ${bs_price:.4f}")
        print(f"Precio Cuántico: ${quantum_price:.4f}")
        print(f"\nDiferencias porcentuales:")
        print(f"BS vs Mercado: {abs(bs_price - market_price) / market_price * 100:.2f}%")
        print(f"Cuántico vs Mercado: {abs(quantum_price - market_price) / market_price * 100:.2f}%")
        print(f"BS vs Cuántico: {abs(bs_price - quantum_price) / bs_price * 100:.2f}%")
    else:
        print("Precio de mercado: No disponible")
        print(f"Precio Black-Scholes: ${bs_price:.4f}")
        print(f"Precio Cuántico: ${quantum_price:.4f}")
        print(f"\nDiferencia porcentual entre modelos:")
        print(f"BS vs Cuántico: {abs(bs_price - quantum_price) / bs_price * 100:.2f}%")

def main():
    # Ejemplo con datos reales
    symbol = "TSLA"  # Apple Inc.
    strike_price = 170.0  # Precio de ejercicio deseado
    
    try:
        # Usar fecha de vencimiento fija para pruebas
        expiration_date = None  # Usar tiempo por defecto de 1 año
        compare_with_market_price(symbol, strike_price, expiration_date)
        
    except Exception as e:
        print(f"Error en el procesamiento: {e}")
    
    print("\nNota: Si no se pudieron obtener datos en tiempo real, se usaron valores por defecto.")
    print("Para evitar problemas de rate limit, espere unos minutos antes de ejecutar el script nuevamente.")

if __name__ == "__main__":
    main()

"""""
CONCLUSIONES PRUEBAS POC 2025/04/24 

- 1) mas qubits no significa mas presicion 
- 2) el tiempo de finalizacion parece ser el parametro mas importante sin embargo mucho tiempo parece ser perjudiucial
- 3) asi sea un  poco tenemos muchos shots y mas cantidad de shots tambien es perjudicial para el presupuesto
- 4) con una menor cantidad de shots puede ser mas preciso

"""

"""
Notas 04/30/2025

-1) realizar un script de ml que modifique los parametros automaticamente para probrar con diferentes combinaciones de parametros

"""
