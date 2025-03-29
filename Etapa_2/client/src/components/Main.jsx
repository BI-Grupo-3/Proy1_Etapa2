import { Tab, Tabs } from '@heroui/react';
import React from 'react';
import { CPUBolt, TuningSquare2 } from 'solar-icon-set';
import DataInstanceForm from './DataInstanceForm';

export default function Main() {
	return (
		<main className='flex flex-col flex-1 p-4 w-full items-center justify-start'>
			<Tabs aria-label='Menu' variant='bordered' className='w-100'>
				<Tab
					key='predict'
					title={
						<div className='flex items-center justify-center gap-2'>
							<CPUBolt iconStyle='Bold' size={20} />
							<span>Predicción</span>
						</div>
					}
					className='w-full'
				>
					<DataInstanceForm isRetrain={false} />
				</Tab>
				<Tab
					key='retrain'
					title={
						<div className='flex items-center justify-center gap-2'>
							<TuningSquare2 iconStyle='Bold' size={20} />
							<span>Reentrenar</span>
						</div>
					}
                    className='w-full'
				>
					<DataInstanceForm isRetrain={true} />
				</Tab>
			</Tabs>
		</main>
	);
}
