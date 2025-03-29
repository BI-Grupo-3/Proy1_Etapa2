import {
	HeroUIProvider,
	Navbar,
	NavbarBrand,
	ToastProvider,
} from '@heroui/react';
import { Incognito } from 'solar-icon-set';
import './App.css';
import Main from './components/Main';

function App() {
	return (
		<HeroUIProvider>
			<ToastProvider placement={'top-center'} toastOffset={60} />
			<Navbar>
				<NavbarBrand className='flex items-center gap-2'>	
					<Incognito
						iconStyle='BoldDuotone'
						size={40}
						color='white'
					/>
					<h1 className='text-2xl font-bold text-white'>
						Detector de Fake News
					</h1>
				</NavbarBrand>
			</Navbar>
			<div className='flex flex-col min-h-screen'>
				<Main />
			</div>
		</HeroUIProvider>
	);
}

export default App;
